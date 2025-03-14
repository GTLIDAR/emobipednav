import logging
import torch
import numpy as np
from numpy.linalg import norm
import itertools
from utils.state import tensor_to_joint_state
from algos.RGL.state_predictor import StatePredictor
from algos.RGL.graph_model import RGL
from algos.RGL.value_estimator import ValueEstimator
from math import hypot, cos, sin

class ModelPredictiveRL:
    def __init__(self, args, time_step, action_range):
        self.name = 'ModelPredictiveRL'
        self.gamma = args.gamma
        self.sampling = args.sampling
        self.speed_samples = args.speed_samples
        self.rotation_samples = args.rotation_samples
        self.rotation_constraint = action_range[1, 1]
        self.v_pref = action_range[1, 0]

        self.time_step = time_step
        
        
        self.epsilon = None
        self.action_values = None
        self.phase = None

        self.robot_state_dim = 9
        self.human_state_dim = 5
        self.sparse_rotation_samples = 8
        
        self.action_group_index = []
        self.traj = None

        graph_model1 = RGL(args, self.robot_state_dim, self.human_state_dim)
        self.value_estimator = ValueEstimator(graph_model1)
        graph_model2 = RGL(args, self.robot_state_dim, self.human_state_dim)
        self.state_predictor = StatePredictor(graph_model2, self.time_step)
        self.model = [graph_model1, graph_model2, self.value_estimator.value_network,
                        self.state_predictor.human_motion_predictor]
        self.state_predictor.time_step = time_step
        self.build_action_space(self.v_pref)

        # LIPM
        self.w = np.sqrt(9.81/1.02)
        self.cosh_wt = np.cosh(self.w * self.time_step)
        self.sinh_wt = np.sinh(self.w * self.time_step)

    def set_phase(self, phase):
        self.phase = phase

    def set_device(self, device):
        self.device = device
        for model in self.model:
            model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
        
    def get_normalized_gamma(self):
        return pow(self.gamma, self.time_step * self.v_pref)

    def get_model(self):
        return self.value_estimator

    def get_state_dict(self):
        if self.state_predictor.trainable:
            return {
                'graph_model1': self.value_estimator.graph_model.state_dict(),
                'graph_model2': self.state_predictor.graph_model.state_dict(),
                'value_network': self.value_estimator.value_network.state_dict(),
                'motion_predictor': self.state_predictor.human_motion_predictor.state_dict()
            }
        else:
            return {
                    'graph_model': self.value_estimator.graph_model.state_dict(),
                    'value_network': self.value_estimator.value_network.state_dict()
                }

    def get_traj(self):
        return self.traj

    def load_state_dict(self, state_dict):
        if self.state_predictor.trainable:
            
            self.value_estimator.graph_model.load_state_dict(state_dict['graph_model1'])
            self.state_predictor.graph_model.load_state_dict(state_dict['graph_model2'])

            self.value_estimator.value_network.load_state_dict(state_dict['value_network'])
            self.state_predictor.human_motion_predictor.load_state_dict(state_dict['motion_predictor'])
        else:
            self.value_estimator.graph_model.load_state_dict(state_dict['graph_model'])
            self.value_estimator.value_network.load_state_dict(state_dict['value_network'])

    def save_model(self, file):
        torch.save(self.get_state_dict(), file)

    def load_model(self, file):
        checkpoint = torch.load(file)
        self.load_state_dict(checkpoint)

    def build_action_space(self, v_pref):
        speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]
        rotations = np.linspace(-self.rotation_constraint, self.rotation_constraint, self.rotation_samples)

        action_space = [np.zeros(2)]
        for j, speed in enumerate(speeds):
            if j == 0:
                # index for action (0, 0)
                self.action_group_index.append(0)
            # only two groups in speeds
            if j < 3:
                speed_index = 0
            else:
                speed_index = 1

            for i, rotation in enumerate(rotations):
                rotation_index = i // 2

                action_index = speed_index * self.sparse_rotation_samples + rotation_index
                self.action_group_index.append(action_index)

                action_space.append(np.array([speed, rotation], dtype=np.float32))

        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    def predict(self, state, last_action):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            max_action = None
            max_value = float('-inf')
            max_traj = None
 
            for action in self.action_space:
                state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
                next_state = self.state_predictor(state_tensor, action, last_action)
                max_next_return, max_next_traj = self.V_planning(next_state)
                reward_est = self.estimate_reward(state, action, last_action)
                value = reward_est + self.get_normalized_gamma() * max_next_return
                if value > max_value:
                    max_value = value
                    max_action = action
                    max_traj = [(state_tensor, action, reward_est)] + max_next_traj
            if max_action is None:
                raise ValueError('Value network is not well trained.')

        if self.phase == 'train':
            self.last_state = self.transform(state)
        else:
            self.traj = max_traj

        return max_action

    def V_planning(self, state):
        """ Plans n steps into future. Computes the value for the current state as well as the trajectories
        defined as a list of (state, action, reward) triples

        """

        current_state_value = self.value_estimator(state)
        return current_state_value, [(state, None, None)]

    
    def is_collision(self, humans, robot_position_radius, discomfort=0.0):
        for i, human in enumerate(humans):
            dis = hypot(robot_position_radius[0] - human.px, robot_position_radius[1] - human.py)
            if dis < robot_position_radius[2] + human.radius + discomfort:
                return True
        return False

    def estimate_reward(self, state, action, last_action):
        """ If the time step is small enough, it's okay to model agent as linear movement during this period

        """
        # collision detection
        if isinstance(state, list) or isinstance(state, tuple):
            state = tensor_to_joint_state(state)
        human_states = state.human_states
        robot_state = state.robot_state

        goal_distance_last = hypot(robot_state.px - robot_state.gx, robot_state.py - robot_state.gy)

        pf_x = (last_action[0] * self.cosh_wt - action[0]) / (self.w * self.sinh_wt)
        x_n =  pf_x - pf_x * self.cosh_wt + last_action[0] * self.sinh_wt / self.w
        robot_theta = robot_state.theta + action[1] * self.time_step
        if robot_theta > np.pi:
            robot_theta -= (2.0 * np.pi)
        elif robot_theta < -np.pi:
            robot_theta += (2.0 * np.pi)
        robot_x = robot_state.px + x_n * cos(robot_theta)
        robot_y = robot_state.py + x_n * sin(robot_theta)
        robot_position_radius = [robot_x, robot_y, robot_state.radius]


        collision = self.is_collision(human_states, robot_position_radius)
        collision_layer = self.is_collision(human_states, robot_position_radius, discomfort=0.2)

        goal_dist = hypot(robot_x - robot_state.gx, robot_y - robot_state.gy)
        dis_goal_reward = 0.3 * (goal_distance_last - goal_dist)

        if self.phase == 'train':
            reaching_goal = goal_dist < (robot_state.radius - 0.2)
        else:
            reaching_goal = goal_dist < (robot_state.radius - 0.1)

        reward = dis_goal_reward + collision_layer * (-0.1)
        if collision:
            reward = -0.6
        elif reaching_goal:
            reward = 0.5

        return reward

    def transform(self, state):
        """
        Take the JointState to tensors

        :param state:
        :return: tensor of shape (# of agent, len(state))
        """
        robot_state_tensor = torch.Tensor([state.robot_state.to_tuple()]).to(self.device)
        human_states_tensor = torch.Tensor([human_state.to_tuple() for human_state in state.human_states]). \
            to(self.device)

        return robot_state_tensor, human_states_tensor
