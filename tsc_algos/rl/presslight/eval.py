'''
@Author: WANG Maonan
@Date: 2026-06-01 01:11:07
@Description: PressLight 评估脚本
-> python eval.py --junction Beijing_Beihuan --env_name normal_increasing_demand \
    --history_len 5 --event_name event_1 --gui
@LastEditTime: 2026-06-25 18:59:39
'''
import sys
import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from junction_configs import load_junction_config, load_event_config
from tsc_algos.rl.presslight.presslight_env.make_env import make_env
from tsc_algos.output_utils import generate_output_paths

path_convert = get_abs_path(__file__)
logger.remove()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PressLight 评估')
    parser.add_argument('--junction', type=str, default='Beijing_Beihuan',
                        help='路口名称')
    parser.add_argument('--env_name', type=str, default='easy_low_density',
                        help='环境名称，如 easy_low_density')
    parser.add_argument('--history_len', type=int, default=5,
                        help='PressLight state/reward 使用的历史帧数，需与训练模型一致')
    parser.add_argument('--reward_time_decay', type=float, default=1.0,
                        help='pressure reward 时间衰减，需与训练模型一致')
    parser.add_argument('--gui', action='store_true', default=False,
                        help='是否开启 SUMO GUI；默认关闭，便于无界面跑出 tripinfo')
    parser.add_argument('--event_name', type=str, default='',
                        help='特殊事件集合名称（定义在 junction_configs 的 EVENTS 中，如 demo）；为空则不注入事件')
    args = parser.parse_args()

    cfg = load_junction_config(args.junction, args.env_name)
    trip_info, fcd_output = generate_output_paths(args.junction, args.env_name, "presslight")

    # 特殊事件配置（来自路口配置文件的 EVENTS 字典）
    accident_configs, special_vehicle_configs = (
        load_event_config(args.junction, args.event_name) if args.event_name else ([], [])
    )

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
        'use_gui': args.gui, # 测试环境打开 GUI 以便观察
        'log_file': log_path,
        'history_len': args.history_len,
        'reward_time_decay': args.reward_time_decay,
        'trip_info': trip_info,
        'fcd_output': fcd_output,
        'accident_configs': accident_configs,
        'special_vehicle_configs': special_vehicle_configs,
    }
    env = DummyVecEnv([make_env(env_index='0', **params)])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = path_convert(f'./checkpoints/{args.junction}_{args.env_name}/last_rl_model.zip')
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
