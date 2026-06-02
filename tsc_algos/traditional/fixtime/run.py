'''
Author: WANG Maonan
Date: 2026-04-13 20:30:54
LastEditTime: 2026-04-14 21:32:05
Description: FixTime 运行入口, 可以控制相位持续时间
-> python run.py --junction Beijing_Beihuan --env_name easy_high_density
'''
import sys
import argparse
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from loguru import logger
from tshub.utils.get_abs_path import get_abs_path

from junction_configs import load_junction_config
from tsc_algos.output_utils import generate_output_paths
from tsc_algos.traditional.fixtime.make_env import make_env
from tsc_algos.traditional.fixtime.fixtime_agent import FixTimeAgent

path_convert = get_abs_path(__file__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FixTime 信号控制')
    parser.add_argument('--junction', type=str, default='Beijing_Beihuan',
                        help='路口名称')
    parser.add_argument('--env_name', type=str, default='easy_low_density',
                        help='环境名称，如 easy_low_density')
    parser.add_argument('--use_gui', action='store_true', default=True,
                        help='是否开启 GUI')
    args = parser.parse_args()

    cfg = load_junction_config(args.junction, args.env_name)

    trip_info, fcd_output = generate_output_paths(
        junction=args.junction, 
        env_name=args.env_name, 
        algo_name="fixtime"
    )

    env = make_env(
        sumo_cfg=cfg['sumo_cfg'], net_file=cfg['net_file'],
        tls_id=cfg['tls_id'], num_seconds=cfg['num_seconds'],
        use_gui=args.use_gui,
        trip_info=trip_info,
        fcd_output=fcd_output,
    )
    agent = FixTimeAgent(
        num_phases=cfg['num_phases'],
        phase_durations=cfg['fix_phase_durations']
    )
    agent.run(env, num_episodes=1)

    # 输出仿真的结果
    logger.info("Simulation Output:")
    logger.info(f"├── Trip Info: {trip_info}")
    logger.info(f"└── FCD Output: {fcd_output}")
