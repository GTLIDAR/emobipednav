import numpy as np
import digit_mujoco.DigitControlPybind.bin._Digit_IFM as DigitControlEnv
from ruamel.yaml import dump, RoundTripDumper


class MBCWrapper:
    def __init__(self, cfg, motor_offset, action_scale):
        self.model_based_controller = None
        self.cfg = cfg
        self.usr_command = None
        # control constants
        self.kp = self.cfg.control.default_kp
        self.kd = self.cfg.control.default_kd
        self.action_scale = action_scale
        self.motor_offset = motor_offset
        # utils for ar2mujoco and vice versa
        self.ar_joint_names = ["left-hip-roll", "left-hip-yaw", "left-hip-pitch", "left-knee", "left-toe-A",
                               "left-toe-B",
                               "right-hip-roll", "right-hip-yaw", "right-hip-pitch", "right-knee", "right-toe-A",
                               "right-toe-B",
                               "left-shoulder-roll", "left-shoulder-pitch", "left-shoulder-yaw", "left-elbow",
                               "right-shoulder-roll", "right-shoulder-pitch", "right-shoulder-yaw", "right-elbow",
                               "left-shin", "left-tarsus", "left-toe-pitch", "left-toe-roll", "left-heel-spring",
                               "right-shin", "right-tarsus", "right-toe-pitch", "right-toe-roll", "right-heel-spring"]
        self.ar_actuator_names = self.ar_joint_names[:20]
        self.ar_passive_hinge_names = self.ar_joint_names[20:]

        self.mujoco_joint_names = ['left-hip-roll', 'left-hip-yaw', 'left-hip-pitch', 'left-achilles-rod', 'left-knee',
                                   'left-shin', 'left-tarsus', 'left-heel-spring',
                                   'left-toe-A', 'left-toe-A-rod', 'left-toe-B', 'left-toe-B-rod', 'left-toe-pitch',
                                   'left-toe-roll',
                                   'left-shoulder-roll', 'left-shoulder-pitch', 'left-shoulder-yaw', 'left-elbow',
                                   'right-hip-roll',
                                   'right-hip-yaw', 'right-hip-pitch', 'right-achilles-rod', 'right-knee', 'right-shin',
                                   'right-tarsus', 'right-heel-spring',
                                   'right-toe-A', 'right-toe-A-rod', 'right-toe-B', 'right-toe-B-rod',
                                   'right-toe-pitch', 'right-toe-roll',
                                   'right-shoulder-roll', 'right-shoulder-pitch', 'right-shoulder-yaw', 'right-elbow']

        self.mujoco_actuator_names = []
        self.mujoco_passive_hinge_names = []
        for joint_name in self.mujoco_joint_names:
            if joint_name in self.ar_passive_hinge_names:
                self.mujoco_passive_hinge_names.append(joint_name)
        for joint_name in self.mujoco_joint_names:
            if joint_name in self.ar_actuator_names:
                self.mujoco_actuator_names.append(joint_name)
        assert len(self.mujoco_actuator_names) == 20

        self.ar2mujoco_order_actuator_list = []
        self.mujoco2ar_order_actuator_list = []
        self.ar2mujoco_order_passive_hinge_list = []
        self.mujoco2ar_order_passive_hinge_list = []

        for key in self.ar_actuator_names:
            self.mujoco2ar_order_actuator_list.append(self.mujoco_actuator_names.index(key))
        for key in self.mujoco_actuator_names:
            self.ar2mujoco_order_actuator_list.append(self.ar_actuator_names.index(key))
        for key in self.ar_passive_hinge_names:
            self.mujoco2ar_order_passive_hinge_list.append(self.mujoco_passive_hinge_names.index(key))
        for key in self.mujoco_passive_hinge_names:
            self.ar2mujoco_order_passive_hinge_list.append(self.ar_passive_hinge_names.index(key))

        # for i in range(20):
        #     print(actuator_names_in_ar_order[self.ar2mujoco_order_actuator_list[i]])
        #     print(mujoco_actuator_names[i])
        # for i in range(20):
        #     print(mujoco_actuator_names[self.mujoco2ar_order_actuator_list[i]])
        #     print(actuator_names_in_ar_order[i])

    def _ar2mujoco_order_actuator(self, motor_list):
        if isinstance(motor_list, list):
            assert len(motor_list) == 20
            motor_list = np.array(motor_list)
            return list(motor_list[self.ar2mujoco_order_actuator_list])
        else:
            assert motor_list.shape == (20,) or motor_list.shape == (1, 20)
            if motor_list.shape == (1, 20):
                return motor_list[0, self.ar2mujoco_order_actuator_list]
            else:
                return motor_list[self.ar2mujoco_order_actuator_list]

    def _mujoco2ar_order_actuator(self, motor_list):
        if isinstance(motor_list, list):
            assert len(motor_list) == 20
            motor_list = np.array(motor_list)
            return list(motor_list[self.mujoco2ar_order_actuator_list])
        else:
            assert motor_list.shape == (20,)
            return motor_list[self.mujoco2ar_order_actuator_list]

    def _ar2mujoco_order_passive_hinge(self, passive_hinge_list):
        if isinstance(passive_hinge_list, list):
            assert len(passive_hinge_list) == 10
            passive_hinge_list = np.array(passive_hinge_list)
            return list(passive_hinge_list[self.ar2mujoco_order_passive_hinge_list])
        else:
            assert passive_hinge_list.shape == (10,)
            return passive_hinge_list[self.ar2mujoco_order_passive_hinge_list]

    def _mujoco2ar_order_passive_hinge(self, passive_hinge_list):
        if isinstance(passive_hinge_list, list):
            assert len(passive_hinge_list) == 10
            passive_hinge_list = np.array(passive_hinge_list)
            return list(passive_hinge_list[self.mujoco2ar_order_passive_hinge_list])
        else:
            assert passive_hinge_list.shape == (10,)
            return passive_hinge_list[self.mujoco2ar_order_passive_hinge_list]

    def _get_ar_digit_state(self, observation):
        observation_copy = observation.copy()
        ar_digit_state = np.zeros((1, 73), dtype=np.float32)
        ar_digit_state[0, 0:13] = observation_copy[0:13]
        ar_digit_state[0, 13:33] = self._mujoco2ar_order_actuator(observation_copy[13:33])
        ar_digit_state[0, 33:43] = self._mujoco2ar_order_passive_hinge(observation_copy[33:43])
        ar_digit_state[0, 43:63] = self._mujoco2ar_order_actuator(observation_copy[43:63])
        ar_digit_state[0, 63:73] = self._mujoco2ar_order_passive_hinge(observation_copy[63:73])
        return ar_digit_state
    
    def set_states(self, obs):
        obs_to_ar = self._get_ar_digit_state(obs.astype(np.float32))
        self.model_based_controller.setStates(obs_to_ar.reshape(1, 73))
        
    def get_domain_switch(self):
        return self.model_based_controller.getDomainSwitch()

    def get_action(self, obs):
        expert_torque = np.zeros((1, 20), dtype=np.float32)
        velreference = np.zeros_like(expert_torque)
        self.model_based_controller.computeTorque(expert_torque, velreference)
        expert_torque = self._ar2mujoco_order_actuator(expert_torque)
        velreference = self._ar2mujoco_order_actuator(velreference)
        if self.cfg.control.control_type == 'PD':
            expert_action = self._change_target_action(expert_torque, obs)
            return np.concatenate((expert_action[:6],expert_action[10:16])).astype(np.float32), expert_torque.astype(np.float32)
        elif self.cfg.control.control_type == 'T':
            return expert_torque

    def _change_target_action(self, expert_torque, observation):
        observation_copy = observation.copy()
        motor_pos = observation_copy[13:33]
        motor_vel = observation_copy[43:63]
        target_joint_pos = (expert_torque + self.kd * motor_vel) / self.kp + motor_pos  # target full joint pos
        # This is in the form of action label(joint diff from nominal pose and action scale)
        return (target_joint_pos - self.motor_offset) / self.action_scale

    def reset(self, obs, usr_command):        
        self.model_based_controller = DigitControlEnv.digitControlEnv(self.cfg.control.control_dt)
        self.model_based_controller.init()
        self.model_based_controller.reset()
        self.set_states(obs)
        self.model_based_controller.setUsrCommand(usr_command.astype(np.float32).reshape(1, 3))
        self.usr_command = usr_command

    def set_command(self, usr_command):
        """ this is only called when environmental usr command is changed without reset call """
        self.model_based_controller.setUsrCommand(usr_command.astype(np.float32).reshape(1, 3))
        self.usr_command = usr_command

    def get_phase_variable(self):
        if self.model_based_controller is None:
            return 0.
        return self.model_based_controller.getPhaseVariable()

    def get_domain(self):
        if self.model_based_controller is None:
            return 0
        domain = self.model_based_controller.getDomain()        
        if domain == 0: # left stand
            return -1
        elif domain == 1: # right stand
            return 1
        elif domain == 2: # dubble suport
            return 0
        else:
            raise ValueError("Domain should be 0, 1, or 2")    
    
