# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from digit_mujoco.cfg.base.base_config import BaseConfig,ConfigObj

class DigitEnvConfig(BaseConfig):
    seed = 0
    class env(ConfigObj):
        max_time = 50.0  # (s), this time should be longer than the time of navigation task
        sim_dt = 0.0005 # (s)
        obs_dim = 52+0
        hist_len_s = 0.1  # (s)
        hist_interval_s = 0.03  # (s).
        act_dim = 12 # target joint pos 

    class terrain(ConfigObj):
        terrain_type = "flat" # flat, test_rough, rough_random???
        terrain_path = "terrain_info"       
    
    class reset_state(ConfigObj):
        random_dof_reset = False
        p_std = 0.03
        v_std = 0.06
        root_p_uniform = 0.4
        root_v_std = 0.1
        random_dof_names = ["left-hip-roll", "left-hip-yaw", "left-hip-pitch", "left-knee",
                            "left-toe-A","left-toe-B",
                            "right-hip-roll", "right-hip-yaw", "right-hip-pitch", "right-knee", 
                            "right-toe-A","right-toe-B",
                            "left-tarsus", "left-toe-pitch", "left-toe-roll",
                            "right-tarsus", "right-toe-pitch", "right-toe-roll"]

    class commands(ConfigObj):
        curriculum = False # when true, vel range should be changed
        max_curriculum = 1.        
        class ranges(ConfigObj):
            # x_vel_range = [0.,1.5] # min max [m/s]
            # y_vel_range = [-0.4,0.4] # min max [m/s]
            # ang_vel_range = [-0.4,0.4] # min max [rad/s]
            x_vel_range = [0.,0.4] # min max [m/s]
            y_vel_range = [-0.2,0.2] # min max [m/s]
            ang_vel_range = [-0.3,0.3] # min max [rad/s]
            
            cut_off = 0.1

    class control(ConfigObj):
        mbc_control = True # if true, mbc action is used in def step()
        control_type = 'PD' # PD: PD control, T: torques
        action_scale = 0.1
        control_dt = 0.005
        lower_motor_index = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
        default_kp = np.array([1400, 1000, 1167, 1300, 533, 533,
                      500, 500, 500, 500,
                      1400, 1000, 1167, 1300, 533, 533,
                      500, 500, 500, 500])
        # default_kp = np.array([1400*30, 1000, 1167*30, 1300*30, 533, 533,
        #               500, 500, 500, 500,
        #               1400*30, 1000, 1167*30, 1300*30, 533, 533,
        #               500, 500, 500, 500])
        default_kd = np.array([5,5,5,5,5,5,
                      5,5,5,5,
                      5,5,5,5,5,5,
                      5,5,5,5])

    class vis_record(ConfigObj):
        visualize = False # should set to false when training
        record = False # should visualize true
        record_fps = 15
        record_env = 0 # -1 if you don't wanna recording. recording should always be done in env 0
        snapshot_gap = 10

    class domain_randomization(ConfigObj):
        is_true = False  
        action_delay = 0.002 # TODO: isn't it to large?
        # friction_noise = [0.4, 2.0] # scaling
        kp_noise = [0.9, 1.1]
        kd_noise = [0.9, 1.1]
        joint_friction = [0., 0.7]

    class rewards(ConfigObj):
        class scales(ConfigObj):
            lin_vel_tracking = 2.
            ang_vel_tracking = 1.5
            z_vel_penalty = -0.01
            roll_pitch_penalty = -0.1
            torque_penalty = -2e-5
            base_orientation_penalty = -0.1
            foot_lateral_distance_penalty = -0.
            swing_foot_fix_penalty = -0.1
            termination = 0.

    class normalization(ConfigObj):
        class obs_scales(ConfigObj):
            lin_vel = 2.0
            ang_vel = 2.0
            dof_pos = 1.0
            dof_vel = 0.05
        clip_obs = 100. # NOTE: clipped action wihtout scaling is included in observation!
        clip_act = 100. # NOTE: make sure to change these when torque control

    class obs_noise(ConfigObj):
        is_true = False
        lin_vel_std = 0.15
        ang_vel_std = 0.15
        dof_pos_std = 0.175
        dof_vel_std = 0.15
        projected_gravity_std = 0.075