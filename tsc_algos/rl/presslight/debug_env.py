'''
@Author: WANG Maonan
@Date: 2026-06-02 13:56:27
@Description: Minimal PressLight env runner for debugging.
@LastEditTime: 2026-06-02 21:43:48
'''
import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from junction_configs import load_junction_config
from tsc_algos.output_utils import generate_output_paths
from tsc_algos.rl.presslight.presslight_env.make_env import make_env
from tsc_algos.rl.presslight.presslight_env.reward_funcs import pressure_reward


path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debug PressLight env.')
    parser.add_argument('--junction', type=str, default='Beijing_Beihuan')
    parser.add_argument('--env_name', type=str, default='normal_fluctuating_commuter')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--reward_scale', type=float, default=1.0)
    parser.add_argument('--history_len', type=int, default=5)
    parser.add_argument('--reward_time_decay', type=float, default=1.0)
    parser.add_argument('--action', type=int, default=None)
    args = parser.parse_args()

    cfg = load_junction_config(args.junction, args.env_name)
    trip_info, fcd_output = generate_output_paths(
        args.junction,
        args.env_name,
        'presslight_debug',
    )

    tsc_env_generate = make_env(
        tls_id=cfg['tls_id'],
        num_phases=cfg['num_phases'],
        sumo_cfg=cfg['sumo_cfg'],
        net_file=cfg['net_file'],
        num_seconds=cfg['num_seconds'],
        use_gui=True,
        log_file=None,
        env_index=0,
        reward_scale=args.reward_scale,
        history_len=args.history_len,
        reward_time_decay=args.reward_time_decay,
        trip_info=trip_info,
        fcd_output=fcd_output,
    )
    tsc_env = tsc_env_generate()

    dones = False
    step_time = 0
    states, reset_infos = tsc_env.reset(seed=args.seed)

    while not dones and step_time < args.steps:
        info_env = tsc_env.env

        # Put breakpoints around these variables when debugging.
        static_lane_features = info_env.static_lane_features
        lane_order = info_env.lane_order
        lane_dynamic_features_seq = info_env.lane_dynamic_features_seq
        tls_dynamic_features_seq = info_env.tls_dynamic_features_seq
        lane_dynamic_features = lane_dynamic_features_seq[-1]
        tls_dynamic_features = tls_dynamic_features_seq[-1]
        unscaled_reward = pressure_reward(
            tls_dynamic_features_seq,
            history_len=args.history_len,
            time_decay=args.reward_time_decay,
        )

        if args.action is None:
            action = np.random.randint(cfg['num_phases'])
        else:
            action = args.action

        states, rewards, truncated, dones, infos = tsc_env.step(action=action)
        logger.info(
            f"SIM: {infos['step_time']} "
            f"Action: {action}; "
            f"StateShape: {states.shape}; "
            f"Reward: {rewards}; "
            f"UnscaledRewardBeforeAction: {unscaled_reward}."
        )
        step_time += 1

    tsc_env.close()
