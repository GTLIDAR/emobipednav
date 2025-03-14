from dowel import tabular
import numpy as np
from digit_mujoco.cfg.base.base_config import ConfigObj
from collections import defaultdict


def convert_cfg_to_dict(cfg):
    cfg_dict = {}
    for attr_name in dir(cfg):
        if not attr_name.startswith('__') and attr_name != 'init_member_classes':
            attr_value = getattr(cfg, attr_name)
            if isinstance(attr_value, ConfigObj):
                attr_dict = convert_cfg_to_dict(attr_value)
                cfg_dict[attr_name] = attr_dict
            else:
                cfg_dict[attr_name] = attr_value
    return cfg_dict

def log_reward_performance(batch, prefix='Evaluation'):
    reward = defaultdict(list)
    for step_reward_dict in batch.env_infos['reward_info']:
        for k,v in step_reward_dict.items():
            reward[k].append(v)
    with tabular.prefix(prefix + '/'):
        for k,v in reward.items():
            tabular.record(k+'_rew',np.mean(v))
        tabular.record('total_rew', np.mean(batch.rewards))
        if batch.env_infos['terrain_level'][0] is not None:
            tabular.record('terrain_level', batch.env_infos['terrain_level'].mean())
