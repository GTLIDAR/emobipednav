import torch
import torch.nn as nn
import numpy as np
from algos.RGL.helpers import mlp
from math import cos, sin

class StatePredictor(nn.Module):
    def __init__(self, graph_model, time_step, X_dim=32, motion_predictor_dims=[64, 5]):
        """
        This function predicts the next state given the current state as input.
        It uses a graph model to encode the state into a latent space and predict each human's next state.
        """
        super().__init__()
        self.trainable = True
        self.graph_model = graph_model
        self.human_motion_predictor = mlp(X_dim, motion_predictor_dims)
        self.time_step = time_step

        # LIPM
        self.w = np.sqrt(9.81/1.02)
        self.cosh_wt = np.cosh(self.w * self.time_step)
        self.sinh_wt = np.sinh(self.w * self.time_step)
        

    def forward(self, state, action, last_action, detach=False):
        """ Predict the next state tensor given current state as input.

        :return: tensor of shape (batch_size, # of agents, feature_size)
        """
        assert len(state[0].shape) == 3
        assert len(state[1].shape) == 3

        state_embedding = self.graph_model(state)
        if detach:
            state_embedding = state_embedding.detach()
        if action is None:
            next_robot_state = None
        else:
            next_robot_state = self.compute_next_state(state[0], action, last_action)
        next_human_states = self.human_motion_predictor(state_embedding)[:, 1:, :]
        next_observation = [next_robot_state, next_human_states]
        return next_observation

    def compute_next_state(self, robot_state, action, last_action):
        # currently it can not perform parallel computation
        if robot_state.shape[0] != 1:
            raise NotImplementedError

        # px, py, vx, vy, radius, gx, gy, v_pref, theta
        next_state = robot_state.clone().squeeze().cpu()

        pf_x = (last_action[0] * self.cosh_wt - action[0]) / (self.w * self.sinh_wt)
        x_n =  pf_x - pf_x * self.cosh_wt + last_action[0] * self.sinh_wt / self.w
        robot_theta = next_state[8] + action[1] * self.time_step
        if robot_theta > np.pi:
            robot_theta -= (2.0 * np.pi)
        elif robot_theta < -np.pi:
            robot_theta += (2.0 * np.pi)
        robot_x = next_state[0] + x_n * cos(robot_theta)
        robot_y = next_state[1] + x_n * sin(robot_theta)

        next_state[8] = robot_theta
        next_state[0] = robot_x
        next_state[1] = robot_y
        next_state[2] = np.cos(next_state[8]) * x_n / self.time_step
        next_state[3] = np.sin(next_state[8]) * x_n / self.time_step

        next_state = next_state.cuda()
        return next_state.unsqueeze(0).unsqueeze(0)