from math import hypot, cos, sin, pi
from utils.state import *

class Robot():
    def __init__(self, radius=0.3, v_pref=0.4):
        self.radius = radius
        self.v_pref = v_pref
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.time_step = None

    def set(self, px, py, gx, gy, vx, vy, theta, v_pref=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        if v_pref is not None:
            self.v_pref = v_pref

    def get_joint_state(self, ob):
        full_state = FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)
        return JointState(full_state, ob)

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_position(self):
        return self.px, self.py

    def compute_pose(self, action, differential=False):
        if differential:
            theta_new = self.theta + action[1] * self.time_step
            if theta_new > pi:
                theta_new -= 2.0 * pi
            elif theta_new < -pi:
                theta_new += 2.0 * pi
            px_new = self.px + action[0] * self.time_step * cos(theta_new)
            py_new = self.py + action[0] * self.time_step * sin(theta_new)
        else:
            px_new = self.px + action[0] * self.time_step 
            py_new = self.py + action[1] * self.time_step 
            theta_new = self.theta
        return px_new, py_new, theta_new

    def get_goal_distance(self):
        return hypot(self.gx - self.px, self.gy- self.py)
    
    def get_sfm_state(self):
        return [self.px, self.py, self.vx, self.vy, self.gx, self.gy, self.v_pref]

    def update_states(self, px, py, theta, action, differential=False):
        """
        Perform an action and update the state
        """
        self.px, self.py, self.theta = px, py, theta
        if differential:
            self.vx = action[0] * cos(theta)
            self.vy = action[0] * sin(theta)
        else:
            self.vx = action[0] 
            self.vy = action[1]

    def update_vel(self, action):
        self.vx = action[0]
        self.vy = action[1]
