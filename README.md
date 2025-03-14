Social navigation with SAC_AE algorithm
#  EmoBipedNav: Emotion-aware Social Navigation for Bipedal Robots with Deep Reinforcement Learning
- [video] (https://youtu.be/fNNL56sTSjY?si=sLgWn5P6o4MndMzl), [project](https://gatech-lidar.github.io/emobipednav.github.io/)
- the networks used in this repo are partially from [sac-ae](https://github.com/denisyarats/pytorch_sac_ae) 

# environment
- Ubuntu 22.04
- NVIDIA GeForce RTX 4090
- Driver Version: 535.171.04   CUDA Version: 12.2
- Anaconda install
- Create Anaconda env, ```conda create -n torch python=3.10```. Do NOT use python 3.12 because pybind11 is not compatible when you compile pybind project.
- ```conda activate torch```
- Install [pytorch](https://pytorch.org/get-started/locally/)


# simulation with simplified LIP model and digit_mujoco
- ```git clone https://github.com/sybrenstuvel/Python-RVO2.git```
- conda activate torch
- pip install Cython
- inside Python-RVO2, ```python setup.py build```, and then ```python setup.py install```.

- inside C_library, ```python setup.py build_ext --inplace```
- in digit_mujoco, ```pip install -e .```, installing low-level controller for Digit
- train and evaluate the model with digit_mujoco, ```python sac_ae_main.py ```.
- test mode, ```python sac_ae_main.py --load_test_model YOUR_MODEL_PATH```, YOUR_MODEL_PATH should be follow the format like step_100000_success_90.
- By default, training, evaluation, and test are all based on Digit in MuJoCo. You can use LIP by changing the parameter robot_model, robot_eval_model, and robot_test_model.

- replay saved trajectory, ```python replay_episode.py```, please revise file path in ./logs/XXX

# benchmark training and testing
- dwa, ```python dwa_main.py```
- drl-vo, ```python drl_vo_main.py```
- lidar-sac, ```python lidar_sac_main.py```
- lndnl, ```python lndnl_main.py```
- using only a single lidar grid map, ```python sac_ae_main.py --frame_stack 1```
- using occupation grid map, ```python sac_ae_main.py --grid_map```
- for the tests of drl-vo, lidar-sac, and lndnl, ```python xxx_main.py --load_test_model YOUR_MODEL_PATH```

# replay episode
- ```python replay_episode.py```, please the the file path inside this script


