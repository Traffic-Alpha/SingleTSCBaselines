'''
Author: WANG Maonan
Date: 2026-04-21 10:31:31
LastEditTime: 2026-04-21 11:00:07
LastEditors: WANG Maonan
Description: 运行 Webster 信号控制算法
'''
import argparse
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tshub.utils.get_abs_path import get_abs_path
from tsc_algos.traditional.webster.make_env import make_env
from tsc_algos.traditional.webster.webster_agent import WebsterAgent
from tsc_algos.output_utils import generate_output_paths
from junction_configs import load_junction_config

path_convert = get_abs_path(__file__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Webster 信号控制')
    parser.add_argument('--junction', type=str, default='Beijing_Beihuan',
                        help='路口名称')
    parser.add_argument('--env_name', type=str, default='easy_low_density',
                        help='环境名称，如 easy_low_density')
    parser.add_argument('--use_gui', action='store_true', default=True,
                        help='是否开启 GUI')
    args = parser.parse_args()

    cfg = load_junction_config(args.junction, args.env_name)

    trip_info, fcd_output = generate_output_paths(args.junction, args.env_name, "webster")

    env = make_env(
        sumo_cfg=cfg['sumo_cfg'],
        net_file=cfg['net_file'],
        tls_id=cfg['tls_id'],
        num_seconds=cfg['num_seconds'],
        use_gui=args.use_gui,
        trip_info=trip_info,
        fcd_output=fcd_output,
    )
    agent = WebsterAgent(num_phases=cfg['num_phases'], max_green_steps=cfg.get('max_green_steps', 12))
    agent.run(env, num_episodes=1)
