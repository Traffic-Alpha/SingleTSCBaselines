'''
@Author: WANG Maonan
@Description: FRAP 评估脚本
'''
import argparse
import torch
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from .env.make_env import make_env
from junction_configs import load_junction_config

path_convert = get_abs_path(__file__)
logger.remove()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FRAP 评估')
    parser.add_argument('--junction', type=str, default='Beijing_Beihuan',
                        help='路口名称')
    parser.add_argument('--env_name', type=str, default='easy_low_density',
                        help='环境名称，如 easy_low_density')
    args = parser.parse_args()

    cfg = load_junction_config(args.junction, args.env_name)

    # #########
    # Init Env
    # #########
    log_path = path_convert('./log/')
    params = {
        'tls_id': cfg['tls_id'],
        'num_seconds': cfg['num_seconds'],
        'num_phases': cfg['num_phases'],
        'sumo_cfg': cfg['sumo_cfg'],
        'net_file': cfg['net_file'],
        'use_gui': True,
        'log_file': log_path,
    }
    env = DummyVecEnv([make_env(env_index='0', **params)])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = path_convert('./models/last_rl_model.zip')
    model = DQN.load(model_path, env=env, device=device)

    # 使用模型进行测试
    obs = env.reset()
    dones = False
    total_reward = 0

    while not dones:
        action, _state = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        total_reward += rewards

    env.close()
    print(f'累积奖励为, {total_reward}.')
