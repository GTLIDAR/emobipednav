import argparse
import numpy as np
import os


from info import *
from env_emobipednav import CrowdSim


# digit in mujoco
from digit_mujoco.cfg.digit_env_config import DigitEnvConfig
from digit_mujoco.envs.digit.digit_env_flat import DigitEnvFlat


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(eval_or_test_env, current_steps, 
                eval_episodes=100, save_directory=None, 
                if_save_video=False, final_test=False):
    avg_reward = 0.
    success_times = []
    collision_times = []
    timeout_times = []
    success = 0
    collision = 0
    timeout = 0
    collision_cases = []
    timeout_cases = []
    if_save_data = False
    seed_all = [23, 24, 37, 38, 42, 47, 49, 59, 61]
    seed_specific = seed_all[0]
    
    for i in range(eval_episodes):
        if_save_data = (i < 10 or final_test)
        if final_test:
            if eval_episodes == 1:
                # please set your own seed to randomize a specific
                
                lidar_image, robot_goal_emotion_state = eval_or_test_env.reset(seed=seed_specific, save_data=if_save_data)
            else:
                lidar_image, robot_goal_emotion_state = eval_or_test_env.reset(seed=i, save_data=if_save_data)
        else:
            lidar_image, robot_goal_emotion_state  = eval_or_test_env.reset(save_data=if_save_data)
        # eval_or_test_env.render()
        # time.sleep(0.2)
        done = False
        ep_step = 0
        
        while ep_step < eval_or_test_env.max_episode_step:
            # t1 = time.time()
            
            action = eval_or_test_env.cal_dwa_action()
            # action = eval_or_test_env.dwa_compute_action()
            lidar_image, robot_goal_state, reward, done, info = eval_or_test_env.step(action, eval=True, save_data=if_save_data)
            # print('time: ', time.time() - t1) # 7.5ms
            # eval_or_test_env.render()
            # time.sleep(0.2)
            avg_reward += reward
            ep_step = ep_step + 1
            if done or ep_step == eval_or_test_env.max_episode_step or isinstance(info, ReachGoal):
                if ep_step == eval_or_test_env.max_episode_step:
                    timeout += 1
                    timeout_cases.append(i)
                    timeout_times.append(eval_or_test_env.time_limit)
                    print('evaluation episode ' + str(i) + ', time out: ' + str(ep_step))
                else:
                    if isinstance(info, ReachGoal):
                        success += 1
                        success_times.append(eval_or_test_env.global_time)
                        print('evaluation episode ' + str(i) + ', goal reaching at evaluation step: ' + str(ep_step))
                    elif isinstance(info, Collision):
                        collision += 1
                        collision_cases.append(i)
                        collision_times.append(eval_or_test_env.global_time)
                        print('evaluation episode ' + str(i) + ', collision occur at evaluation step: ' + str(ep_step))       
                    else:
                        raise ValueError('Invalid end signal from environment')
                break
        if save_directory is not None:
            if not isinstance(info, ReachGoal):
                if eval_episodes == 1:
                    file_name = save_directory + '/eval_' + str(seed_specific) + '_' + str(i) + '_fail' + '.npz'
                else:
                    file_name = save_directory + '/eval_' + str(current_steps) + '_' + str(i) + '_fail' + '.npz'
            else:
                if eval_episodes == 1:
                    file_name = save_directory + '/eval_' + str(seed_specific) + '_' + str(i) + '.npz'
                else:
                    file_name = save_directory + '/eval_' + str(current_steps) + '_' + str(i) + '.npz'
            file_name_digit = save_directory + '/eval_' + str(seed_specific) + '_' + str(i) + '_digit' + '.npy'
            if if_save_data:
                np.savez_compressed(file_name, **eval_or_test_env.log_env)
                if eval_episodes == 1:
                    digit_qpos_arr = np.array(eval_or_test_env.digit_qpos)
                    np.save(file_name_digit, digit_qpos_arr)

    success_rate = success / eval_episodes
    collision_rate = collision / eval_episodes
    assert success + collision + timeout == eval_episodes
    avg_nav_time = sum(success_times) / len(success_times) if success_times else eval_or_test_env.time_limit

    
    return success_rate, collision_rate, avg_nav_time


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--policy", default="dwa")
    # options, lip, digit_mujoco, now digit_arsim is not supported for training
    parser.add_argument("--robot_model", default="digit_mujoco")
    # options, lip, digit_mujoco, digit_arsim, and digit_hardware
    parser.add_argument("--robot_test_model", default="digit_mujoco")
    parser.add_argument('--use_angular', action='store_true', default=False)
    parser.add_argument('--test_single', action='store_true', default=False)
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=1, type=int)
   
    # environment settings
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--lidar_dim", type=int, default=1800)
    parser.add_argument("--lidar_feature_dim", type=int, default=50)
    # lidar to image
    parser.add_argument('--image_size', default=100, type=int)
    parser.add_argument('--frame_stack', default=9, type=int)
    parser.add_argument('--grid_map', action='store_true', default=False)
    # 2 robot speed, 2 local goal
    parser.add_argument("--robot_goal_state_dim", type=int, default=4)
    parser.add_argument("--laser_angle_resolute", type=float, default=0.003490659)
    parser.add_argument("--laser_min_range", type=float, default=0.27)
    parser.add_argument("--laser_max_range", type=float, default=6.0)
    parser.add_argument("--square_width", type=float, default=10.0)
    parser.add_argument("--human_num_max", type=int, default=4)
    parser.add_argument("--static_obstacle_num_max", type=int, default=3)
    args = parser.parse_args()

    print("---------------------------------------")
    print(f"Policy: {args.policy}, RobotModel: {args.robot_model}, Seed: {args.seed}")
    print("---------------------------------------")
    
    file_prefix = './logs/' + args.policy + '_' + args.robot_model 
   
   
    file_prefix = file_prefix + '/' + '/seed_' + str(args.seed)  
    file_results = file_prefix + '/results'
    file_models = file_prefix + '/models'
    file_evaluation_episodes = file_prefix + '/evaluation_episodes'
    file_final_test_episodes = file_prefix + '/final_test_episodes_digit_joint'
    file_buffer = file_prefix + '/buffer'

    if not os.path.exists(file_results):
        os.makedirs(file_results)

    if not os.path.exists(file_models):
        os.makedirs(file_models)

    if not os.path.exists(file_evaluation_episodes):
        os.makedirs(file_evaluation_episodes)
        
    if not os.path.exists(file_final_test_episodes):
        os.makedirs(file_final_test_episodes)
        
    if not os.path.exists(file_buffer):
        os.makedirs(file_buffer)

    action_range = np.array([[0.0, -0.5],
                             [0.4,  0.5]])
    
    action_resolution = [0.05, 0.1]
    action_num = [int((action_range[1, 0] - action_range[0, 0]) / action_resolution[0]) + 1,
                  int((action_range[1, 1] - action_range[0, 1]) / action_resolution[1]) + 1]
    action_choice_1 = np.linspace(action_range[0, 0], action_range[1, 0], num=action_num[0])
    action_choice_2 = np.linspace(action_range[0, 1], action_range[1, 1], num=action_num[1])
    action_choice_dim = action_num[0] * action_num[1]
    action_choices = np.zeros((action_num[0], action_num[1], 2), dtype=np.float32)
    for i in range(action_num[0]):
        for j in range(action_num[1]):
            action_choices[i, j, :] = np.array([action_choice_1[i], action_choice_2[j]]) 
    action_choices = np.reshape(action_choices, (action_choice_dim, 2))
    
    if args.robot_test_model == 'digit_mujoco':
        cfg_digit_env_test = DigitEnvConfig()
        cfg_digit_env_test.vis_record.visualize = True
        cfg_digit_env_test.vis_record.record = True
        digit_env_test = DigitEnvFlat(cfg_digit_env_test, file_final_test_episodes)
        if_save_video_test = True
    else:
        digit_env_test = None
        if_save_video_test = False
    
    
    test_env = CrowdSim(args, action_range, action_choices=action_choices, digit_env=digit_env_test)

         
    print('dwa test')
    if args.test_single:
        test_times = 1
        current_step = 9999
    else:
        test_times = 500
        current_step = 0
    success_rate, collision_rate, avg_nav_time = eval_policy(test_env, current_step, 
                                                             eval_episodes=test_times, save_directory=file_final_test_episodes, 
                                                             if_save_video=if_save_video_test, final_test=True)

    print('success_rate, collision_rate, avg_nav_time')
    print(success_rate, collision_rate, avg_nav_time)


if __name__ == "__main__":
    main()
