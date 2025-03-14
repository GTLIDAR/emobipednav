import datetime
import os

import akro
import numpy as np
import cv2

import transforms3d as tf3

import mujoco
import mujoco_viewer

from digit_mujoco.utils.reward_functions import *
from gym import utils
from math import fabs


class DigitEnvBase:
    def __init__(self, cfg, log_dir=""):
        self.home_path = os.path.dirname(os.path.realpath(__file__)) + "/../.."
        self.log_dir = log_dir

        # config        
        self.cfg = cfg
        
        # constants
        self.num_substeps = int(self.cfg.control.control_dt / (self.cfg.env.sim_dt + 1e-8)) + 1
        self.record_interval = int(1 / (self.cfg.control.control_dt * self.cfg.vis_record.record_fps))

        self.history_len = int(self.cfg.env.hist_len_s / self.cfg.control.control_dt)
        self.hist_interval = int(self.cfg.env.hist_interval_s / self.cfg.control.control_dt)      

        # control constants that can be changed with DR
        self.kp = self.cfg.control.default_kp
        self.kd = self.cfg.control.default_kd
        self.default_geom_friction = None
        self.motor_joint_friction = np.zeros(20)
        self.usr_command = np.zeros(3, dtype=np.float32)

        # action has nan
        self.action_has_nan = False
        
        # containers (should be reset in reset())    
        self.action = None # only lower body
        self.full_action = None # full body
        self.step_cnt = None

        self.joint_pos_hist = None
        self.joint_vel_hist = None
        # assert self.cfg.env.obs_dim - 70 == int(self.history_len / self.hist_interval) * 12 * 2 # lower motor joints only

        # containers (should be set at the _post_physics_step())        
        self.terminal = None

        # containters (should be set at the _get_obs())
        self.actor_obs = None
        self.robot_state = None
        self.root_quat = np.zeros(4)
        self.root_rpy = np.zeros(3)
        self.root_xy_pos = np.zeros(2)
        self.root_world_height = 1.1

        # containers (should be initialized in child classes)
        self.interface = None
        self.nominal_qvel = None
        self.nominal_qpos = None
        self.nominal_motor_offset = None
        self.model = None
        self.data = None
        self._mbc = None     
        self.domain_switch = False   

        self.curr_terrain_level = None
        self._observation_space = akro.Box(low=-self.cfg.normalization.clip_obs,
                                           high=self.cfg.normalization.clip_obs,
                                           shape=(self.cfg.env.obs_dim,),
                                           dtype=np.float32)
        self._action_space = akro.Box(low=-self.cfg.normalization.clip_act,
                                      high=self.cfg.normalization.clip_act,
                                      shape=(self.cfg.env.act_dim,),
                                      dtype=np.float32)
   
    @property
    def action_space(self):
        """akro.Space: The action space specification."""
        return self._action_space

    @property
    def observation_space(self):
        """akro.Space: The observation space specification."""
        return self._observation_space

    
    def reset(self, robot=None):
        # domain randomization
        if self.cfg.domain_randomization.is_true:
            self.domain_randomization()
        # TODO: debug everything here
        # reset containers
        self.step_cnt = 0
        self.terminal = False
        self.frames = []

        # reset containers that are used in _get_obs
        self.joint_pos_hist = [np.zeros(12)] * self.history_len
        self.joint_vel_hist = [np.zeros(12)] * self.history_len
        self.action = np.zeros(self.cfg.env.act_dim, dtype=np.float32)
        self.action_has_nan = False

        self.usr_command = np.zeros(3, dtype=np.float32)

        # setstate for initialization
        self._reset_state(robot=robot)

        # observe for next step
        self._get_obs() # call _reset_state before this.

        # start rendering              
        if self.viewer is not None and self.cfg.vis_record.visualize:
            frame = self.render()
            if frame is not None:
                self.frames.append(frame)
        
        # reset mbc
        self._do_extra_in_reset() # self.robot_state should be updated before this by calling _get_obs
        self._step_assertion_check()
        
        # second return is for episodic info
        # not sure if copy is needed but to make sure...
        return self.get_eps_info()
    
    def step(self, action):
        if self.step_cnt is None:
            raise RuntimeError('reset() must be called before step()!')
        assert np.all(self._mbc.usr_command == self.usr_command)
        self._mbc_action, self._mbc_torque = self._mbc.get_action(self.robot_state) # this should be called to update phase var and domain  
        if self.cfg.control.mbc_control:
            if self.cfg.control.control_type == 'PD':
                action = self._mbc_action.astype(np.float32)
            elif self.cfg.control.control_type == 'T':
                action = self._mbc_torque.astype(np.float32)
            if np.any(np.isnan(action)):
                self.action_has_nan = True
                action = np.nan_to_num(action, nan=self.action_space.high)

        # clip action
        self.action = np.clip(action, self.action_space.low, self.action_space.high)
        # action space is only leg. actual motor inlcudes upper body.
        self.full_action = np.concatenate((self.action[:6], np.zeros(4), self.action[6:], np.zeros(4))) 

        # control step
        if self.cfg.control.control_type == 'PD':
            target_joint = self.full_action * self.cfg.control.action_scale + self.nominal_motor_offset
            if self.cfg.domain_randomization.is_true:
                self.action_delay_time = int(np.random.uniform(0, self.cfg.domain_randomization.action_delay,1) / self.cfg.env.sim_dt)
            self._pd_control(target_joint, np.zeros_like(target_joint))
        if self.cfg.control.control_type == 'T':
            self._torque_control(self.full_action)

        rewards, tot_reward = self._post_physics_step()
        
        self._mbc.set_states(self.robot_state)
        self.domain_switch = self._mbc.get_domain_switch()
        
        return {'action': self.action, # clipped unscaled action
                'reward': tot_reward,
                'observation': self.actor_obs.copy(),  # this observation is next state
                'env_info': {
                    'reward_info': rewards,
                    'crazy_digit': self.terminal,
                    'curr_value_obs': None,
                    'robot_state': self.robot_state.copy(),
                    'action_label': np.zeros_like(self._mbc_action) if np.isnan(self._mbc_action).any() else self._mbc_action, # this should be given in worker.
                    'terrain_level': self.curr_terrain_level,
                    # these are for testing. make sure the command is not resampled in _post_physics_step
                    "tracking_errors":{
                    'x_vel_error': abs(self.root_lin_vel[0] - self.usr_command[0]),
                    'y_vel_error': abs(self.root_lin_vel[1] - self.usr_command[1]),
                    'ang_vel_error': abs(self.root_ang_vel[2] - self.usr_command[2]),
                    "z_vel_error": abs(self.root_lin_vel[2]),
                    "roll_vel_error": abs(self.root_ang_vel[0]),
                    "pitch_vel_error": abs(self.root_ang_vel[1])},
                    }   
                }
    
    def set_obstacles(self, obstacles):
        for i in range(obstacles.shape[0]):
            self.data.qpos[61+i*7] = obstacles[i][0]
            self.data.qpos[61+i*7+1] = obstacles[i][1]
        mujoco.mj_forward(self.model, self.data)
    
    def get_eps_info(self):
        """
        return current environment's info.
        These informations are used when starting the episodes. starting obeservations.
        """
        return self.actor_obs.copy()
    

    def set_vel_command(self, command):
        """
        command is a dictionary and in the robot frame
        """
        self.usr_command = np.array([command['x_vel'] , command['y_vel'], command['yaw_vel']])
        self._mbc.set_command(self.usr_command)

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def _reset_state(self, robot=None):
        raise NotImplementedError
    
    def _torque_control(self, torque):
        ratio = self.interface.get_gear_ratios().copy()
        for _ in range(self._num_substeps): # this is open loop torque control. no feedback.
            tau = [(i / j) for i, j in zip(torque, ratio)] # TODO: why divide by ratio..? This need to be checked
            self.interface.set_motor_torque(tau)
            self.interface.step()

    def _pd_control(self, target_pos, target_vel):
        self.interface.set_pd_gains(self.kp, self.kd)
        ratio = self.interface.get_gear_ratios().copy()
        for cnt in range(self.num_substeps): # this is PD feedback loop
            if self.cfg.domain_randomization.is_true:
                motor_vel = self.interface.get_act_joint_velocities()
                motor_joint_friction = self.motor_joint_friction * np.sign(motor_vel)
                if cnt < self.action_delay_time:
                    tau = motor_joint_friction
                    tau = [(i / j) for i, j in zip(tau, ratio)]
                    self.interface.set_motor_torque(tau)
                    self.interface.step()
                else:                    
                    tau = self.interface.step_pd(target_pos, target_vel)
                    tau += motor_joint_friction
                    tau = [(i / j) for i, j in zip(tau, ratio)]
                    self.interface.set_motor_torque(tau)
                    self.interface.step()
            else:
                tau = self.interface.step_pd(target_pos, target_vel) # this tau is joint space torque
                tau = [(i / j) for i, j in zip(tau, ratio)]  
                self.interface.set_motor_torque(tau)
                self.interface.step()

    def _post_physics_step(self):
        # observe for next step
        self._get_obs()
        self._is_terminal()

        # TODO: debug reward function
        rewards, tot_reward = self._compute_reward()

        # visualize
        if self.viewer is not None and self.step_cnt % self.record_interval == 0 and self.cfg.vis_record.visualize and self.step_cnt:
            frame = self.render()
            if frame is not None:
                self.frames.append(frame)

        self.step_cnt += 1
  
        if self.terminal:
            self.step_cnt = 0  # this becomes zero when reset is called  

        return rewards, tot_reward
    
    def _get_obs(self):
        """" 
        update actor_obs, robot_state, all the other states
        make sure to call _reset_state before this.
        self._mbc is not reset when first call but it is okay for self._mbc.get_phase_variable(), self._mbc.get_domain(). check those functions.
        """
        # TODO: check all the values
        # update states
        self.qpos = self.data.qpos.copy()
        self.qvel = self.data.qvel.copy()
        self._update_root_state()
        self._update_joint_state()
        self._update_joint_hist()
        self._update_robot_state()
        
        # update observations
        self.projected_gravity = self.interface.get_projected_gravity_vec()
        self.noisy_projected_gravity = self.projected_gravity + np.random.normal(0, self.cfg.obs_noise.projected_gravity_std, 3)
        self.noisy_projected_gravity = self.noisy_projected_gravity / np.linalg.norm(self.noisy_projected_gravity)
        self._update_actor_obs()

        # not sure if copy is needed but to make sure...        

    def _update_root_state(self):
        # root states
        self.root_xy_pos = self.qpos[0:2]
        self.root_world_height = self.qpos[2]
        self.root_quat = self.qpos[3:7]
        roll, pitch, yaw = tf3.euler.quat2euler(self.root_quat, axes='sxyz')
        self.root_rpy = np.array([roll, pitch, yaw])
        base_rot = tf3.euler.euler2mat(0, 0, yaw, 'sxyz')
        self.root_lin_vel = np.transpose(base_rot).dot(self.qvel[0:3]) # world to local
        # do not rotate the world angular velocities into local frame
        # otherwise, the wbc will have a lethal rotation problem. 
        # The digit robot falls when yaw is more than 90 degrees.
        self.root_ang_vel = self.qvel[3:6] 
        self.noisy_root_ang_vel = self.root_ang_vel + np.random.normal(0, self.cfg.obs_noise.ang_vel_std, 3)
        self.noisy_root_lin_vel = self.root_lin_vel + np.random.normal(0, self.cfg.obs_noise.lin_vel_std, 3)
    
    def _update_joint_state(self):
        # motor states
        self.motor_pos = self.interface.get_act_joint_positions()
        self.motor_vel = self.interface.get_act_joint_velocities()
        # passive hinge states
        self.passive_hinge_pos = self.interface.get_passive_hinge_positions()
        self.passive_hinge_vel = self.interface.get_passive_hinge_velocities()

        self.noisy_motor_pos = self.motor_pos + np.random.normal(0, self.cfg.obs_noise.dof_pos_std, 20)
        self.noisy_motor_vel = self.motor_vel + np.random.normal(0, self.cfg.obs_noise.dof_vel_std, 20)
        self.noisy_passive_hinge_pos = self.passive_hinge_pos + np.random.normal(0, self.cfg.obs_noise.dof_pos_std, 10)
        self.noisy_passive_hinge_vel = self.passive_hinge_vel + np.random.normal(0, self.cfg.obs_noise.dof_vel_std, 10)


    def _update_joint_hist(self):
        # joint his buffer update
        self.joint_pos_hist.pop(0)
        self.joint_vel_hist.pop(0)
        if self.cfg.obs_noise.is_true:
            self.joint_pos_hist.append(np.array(self.noisy_motor_pos)[self.cfg.control.lower_motor_index])
            self.joint_vel_hist.append(np.array(self.noisy_motor_vel)[self.cfg.control.lower_motor_index])
        else:
            self.joint_pos_hist.append(np.array(self.motor_pos)[self.cfg.control.lower_motor_index])
            self.joint_vel_hist.append(np.array(self.motor_vel)[self.cfg.control.lower_motor_index])
        assert len(self.joint_vel_hist) == self.history_len

        # assign joint history obs
        self.joint_pos_hist_obs = []
        self.joint_vel_hist_obs = []
        for i in range(int(self.history_len/self.hist_interval)):
            self.joint_pos_hist_obs.append(self.joint_pos_hist[i*self.hist_interval])
            self.joint_vel_hist_obs.append(self.joint_vel_hist[i*self.hist_interval])
        assert len(self.joint_pos_hist_obs) == 3
        self.joint_pos_hist_obs = np.concatenate(self.joint_pos_hist_obs).flatten()
        self.joint_vel_hist_obs = np.concatenate(self.joint_vel_hist_obs).flatten()

    def _update_robot_state(self):
        ''' robot state is state used for MBC '''
        # body_height = self.root_world_height - self._get_height(self.root_xy_pos[0] , self.root_xy_pos[1])
        body_height = self.root_world_height
        root_pos = np.array([self.root_xy_pos[0], self.root_xy_pos[1], body_height])
        self.robot_state = np.concatenate([
            root_pos,  # 2 0~3
            self.root_quat,  # 4 3~7
            self.root_lin_vel,  # 3 7~10
            self.root_ang_vel,  # 3 10~13
            self.motor_pos,  # 20 13~33
            self.passive_hinge_pos,  # 10 33~43
            self.motor_vel,  # 20     43~63
            self.passive_hinge_vel  # 10 63~73
        ])

    def _update_actor_obs(self):
        # NOTE: make sure to call get_action from self._mbc so that phase_variable is updated
        # if self.cfg.obs_noise.is_true:
        #     self.actor_obs = np.concatenate([self.noisy_projected_gravity, # 3
        #                                      self.noisy_root_lin_vel,
        #                                      self.noisy_root_ang_vel,
        #                                      np.array(self.noisy_motor_pos)[self.cfg.control.lower_motor_index] * self.cfg.normalization.obs_scales.dof_pos, # 12
        #                                      np.array(self.noisy_passive_hinge_pos) * self.cfg.normalization.obs_scales.dof_pos, # 10
        #                                      np.array(self.noisy_motor_vel)[self.cfg.control.lower_motor_index] * self.cfg.normalization.obs_scales.dof_vel, # 12
        #                                      np.array(self.noisy_passive_hinge_vel) * self.cfg.normalization.obs_scales.dof_vel, # 10
        #                                      np.array([self._mbc.get_phase_variable(), self._mbc.get_domain()]), # 2
        #                                      self.joint_pos_hist_obs * self.cfg.normalization.obs_scales.dof_pos,
        #                                      self.joint_vel_hist_obs * self.cfg.normalization.obs_scales.dof_vel]).astype(np.float32).flatten()
        # else:
        #     self.actor_obs = np.concatenate([self.projected_gravity, # 3
        #                                      self.root_lin_vel,
        #                                      self.root_ang_vel,
        #                                      np.array(self.motor_pos)[self.cfg.control.lower_motor_index] * self.cfg.normalization.obs_scales.dof_pos, # 12
        #                                      np.array(self.passive_hinge_pos) * self.cfg.normalization.obs_scales.dof_pos, # 10
        #                                      np.array(self.motor_vel)[self.cfg.control.lower_motor_index] * self.cfg.normalization.obs_scales.dof_vel, # 12
        #                                      np.array(self.passive_hinge_vel) * self.cfg.normalization.obs_scales.dof_vel, # 10
        #                                      np.array([self._mbc.get_phase_variable(), self._mbc.get_domain()]), # 2
        #                                      self.joint_pos_hist_obs * self.cfg.normalization.obs_scales.dof_pos,
        #                                      self.joint_vel_hist_obs * self.cfg.normalization.obs_scales.dof_vel]).astype(np.float32).flatten()   
        if self.cfg.obs_noise.is_true:
            self.actor_obs = np.concatenate([self.noisy_root_lin_vel,
                                             self.noisy_root_ang_vel,
                                             np.array(self.noisy_motor_pos)[self.cfg.control.lower_motor_index] * self.cfg.normalization.obs_scales.dof_pos, # 12
                                             np.array(self.noisy_passive_hinge_pos) * self.cfg.normalization.obs_scales.dof_pos, # 10
                                             np.array(self.noisy_motor_vel)[self.cfg.control.lower_motor_index] * self.cfg.normalization.obs_scales.dof_vel, # 12
                                             np.array(self.noisy_passive_hinge_vel) * self.cfg.normalization.obs_scales.dof_vel, # 10
                                             np.array([self._mbc.get_phase_variable(), self._mbc.get_domain()]) # 2
                                             ]).astype(np.float32).flatten()
        else:
            self.actor_obs = np.concatenate([self.root_lin_vel,
                                             self.root_ang_vel,
                                             np.array(self.motor_pos)[self.cfg.control.lower_motor_index] * self.cfg.normalization.obs_scales.dof_pos, # 12
                                             np.array(self.passive_hinge_pos) * self.cfg.normalization.obs_scales.dof_pos, # 10
                                             np.array(self.motor_vel)[self.cfg.control.lower_motor_index] * self.cfg.normalization.obs_scales.dof_vel, # 12
                                             np.array(self.passive_hinge_vel) * self.cfg.normalization.obs_scales.dof_vel, # 10
                                             np.array([self._mbc.get_phase_variable(), self._mbc.get_domain()]) # 2
                                             ]).astype(np.float32).flatten()         
        assert self.actor_obs.shape[0] == self.cfg.env.obs_dim
    
    def _do_extra_in_reset(self):
        self._mbc.reset(self.robot_state, self.usr_command)  # get_obs should be called before this to update robot_state
        self.domain_switch = self._mbc.get_domain_switch()

    def _step_assertion_check(self):
        assert self._mbc.get_phase_variable() == 0.
        assert self._mbc.get_domain() == 1 # right leg
        assert self.usr_command is not None
        assert self._mbc.usr_command is not None

    def _check_nan_action(self):
        delta_q = self.data.qpos[self.get_motor_qposadr()] - self.nominal_qpos[self.get_motor_qposadr()]
        for value in delta_q:
            if fabs(value) > np.pi / 6.0:
                return True
        return False
    
    def _is_terminal(self):
        # self_collision_check = self.interface.check_self_collisions()
        # bad_collision_check = self.interface.check_bad_collisions()
        # lean_check = self.interface.check_body_lean()  # TODO: no lean when RL training. why...?
        # terminate_conditions = {"self_collision_check": self_collision_check,
        #                         "bad_collision_check": bad_collision_check,
        #                         # "body_lean_check": lean_check,
        #                         }
        
        root_vel_crazy_check = (fabs(self.root_lin_vel[0]) > 1.5) or (fabs(self.root_lin_vel[1]) > 1.5) or (fabs(self.root_lin_vel[2]) > 1.0) # as in digit controller
        self_collision_check = self.interface.check_self_collisions()
        robot_obstacle_collision_check = self.interface.check_robot_obstacle_collisions()
        body_lean_check = self.interface.check_body_lean()
        mbc_divergence_check = np.isnan(self._mbc_torque).any() or np.isnan(self._mbc_action).any() #TODO:remove this when RL.
        terminate_conditions = {"root_vel_crazy_check": root_vel_crazy_check,
                                # "self_collision_check": self_collision_check,
                                "robot_obstacle_collision_check": robot_obstacle_collision_check,
                                "action_has_nan_check": self.action_has_nan, 
                                "body_lean_check": body_lean_check,
                                # "mbc_divergence_check": mbc_divergence_check}
        }

        self.terminal = True in terminate_conditions.values()

    def _compute_reward(self):
        # the states are after stepping.
        lin_vel_tracking_reward = lin_vel_tracking(self.root_lin_vel, self.usr_command)        
        ang_vel_tracking_reward = ang_vel_tracking(self.root_ang_vel, self.usr_command)
        z_vel_penalty_reward = z_vel_penalty(self.root_lin_vel)
        roll_pitch_penalty_reward = roll_pitch_penalty(self.root_ang_vel)
        base_orientation_penalty_reward = base_orientation_penalty(self.projected_gravity)
        torque = np.array(self.interface.get_act_joint_torques())[self.cfg.control.lower_motor_index]
        torque_penalty_reward = torque_penalty(torque)

        rfoot_pose = np.array(self.interface.get_rfoot_keypoint_pos()).T
        lfoot_pose = np.array(self.interface.get_lfoot_keypoint_pos()).T
        rfoot_pose = self.interface.change_positions_to_rotated_world_frame(rfoot_pose)
        lfoot_pose = self.interface.change_positions_to_rotated_world_frame(lfoot_pose)

        foot_lateral_distance_penalty_reward = 1.0 if foot_lateral_distance_penalty(rfoot_pose, lfoot_pose) else 0.

        rfoot_grf = self.interface.get_rfoot_grf()
        lfoot_grf = self.interface.get_lfoot_grf()        

        swing_foot_fix_penalty_reward =  swing_foot_fix_penalty(lfoot_grf, rfoot_grf, self.action)

        termination_reward = 1. if self.terminal else 0.

        rewards_tmp = {"lin_vel_tracking": lin_vel_tracking_reward,
                       "ang_vel_tracking": ang_vel_tracking_reward,
                       "z_vel_penalty": z_vel_penalty_reward,
                       "roll_pitch_penalty": roll_pitch_penalty_reward,
                       "base_orientation_penalty": base_orientation_penalty_reward,
                       "torque_penalty": torque_penalty_reward,
                       "foot_lateral_distance_penalty": foot_lateral_distance_penalty_reward,
                       "swing_foot_fix_penalty": swing_foot_fix_penalty_reward,
                       "termination": termination_reward,
                       }        
        rewards = {}
        tot_reward = 0.
        for key in rewards_tmp.keys():
            rewards[key] = getattr(self.cfg.rewards.scales,key) * rewards_tmp[key]
            tot_reward += rewards[key]                

        return rewards, tot_reward

    """
    Visualization Code
    """
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def visualize(self):
        """Creates a visualization of the environment."""
        assert self.cfg.vis_record.visualize, 'you should set visualize flag to true'
        assert self.viewer is None, 'there is another viewer'
        # if self.viewer is not None:
        #     #     self.viewer.close()
        #     #     self.viewer = None
        #     return
        if self.cfg.vis_record.record:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, 'offscreen')
        else:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer_setup()

    def viewer_setup(self):
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # Set camera type to free
        self.viewer.cam.trackbodyid = -1  # Set the body to track (use -1 to track no body)
        self.viewer.cam.distance = 20.0  # Increase distance for a broader view
        self.viewer.cam.elevation = -20  # Adjust elevation for a better perspective
        self.viewer.cam.lookat[0] = 0  # Look at the center of the environment
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0


    def viewer_is_paused(self):
        return self.viewer._paused
    
    def render(self):       
        assert self.viewer is not None

        if self.cfg.vis_record.record:
            return self.viewer.read_pixels(camid=0)
        else:
            self.viewer.render()
            return None
    
    def save_video(self, name, current_step=-1):
        assert self.cfg.vis_record.record
        assert self.log_dir is not None
        assert self.viewer is not None
        if current_step >= 0:
            video_dir = os.path.join(self.log_dir, "video_" + str(current_step))
        else:
            video_dir = os.path.join(self.log_dir, "video")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, name + ".mp4")
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_writer = cv2.VideoWriter(video_path, fourcc, self.cfg.vis_record.record_fps,
                                       (self.frames[0].shape[1], self.frames[0].shape[0]))
        for frame in self.frames:
            frame_switch = np.zeros_like(frame)
            frame_switch[:,:, 0] = frame[:, :, 2]
            frame_switch[:,:, 1] = frame[:, :, 1]
            frame_switch[:,:, 2] = frame[:, :, 0]
            video_writer.write(frame_switch)
        video_writer.release()
        self.frames = []

    def domain_randomization(self):
        # NOTE: the parameters in mjModel shouldn't be changed in runtime!
        # self.model.geom_friction[:,0] = self.default_geom_friction[:,0] * np.random.uniform(self.cfg.domain_randomization.friction_noise[0],
        #                                                                           self.cfg.domain_randomization.friction_noise[1],
        #                                                                           size=self.default_geom_friction[:,0].shape)
        self.motor_joint_friction = np.random.uniform(self.cfg.domain_randomization.joint_friction[0],
                                                        self.cfg.domain_randomization.joint_friction[1],
                                                        size=self.motor_joint_friction.shape)
        self.kp = self.cfg.control.default_kp * np.random.uniform(self.cfg.domain_randomization.kp_noise[0], self.cfg.domain_randomization.kp_noise[1], size=self.cfg.control.default_kp.shape)
        self.kd = self.cfg.control.default_kd * np.random.uniform(self.cfg.domain_randomization.kd_noise[0], self.cfg.domain_randomization.kd_noise[1], size=self.cfg.control.default_kd.shape)
        # self.model.dof_frictionloss[:] = self._default_frictionloss * np.random.uniform(1-self._domain_noise['joint_friction_noise'][0],
    #                                                                                     1+self._domain_noise['joint_friction_noise'][1],
    #                                                                                     size=self._default_frictionloss.shape)