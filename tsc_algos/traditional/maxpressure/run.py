'''
Author: WANG Maonan
Date: 2026-04-21 10:35:14
@LastEditTime: 2026-06-26 00:01:26
@LastEditors: WANG Maonan
Description: 运行 MaxPressure 获取仿真结果
-> python run.py --junction Beijing_Beihuan --env_name normal_fluctuating_commuter --use_gui --min_green_steps 1 --max_green_steps 12
'''
import sys
import argparse
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tshub.utils.get_abs_path import get_abs_path
from tsc_algos.traditional.maxpressure.make_env import make_env
from tsc_algos.traditional.maxpressure.maxpressure_agent import MaxPressureAgent
from tsc_algos.output_utils import generate_output_paths
from junction_configs import load_junction_config, load_event_config

path_convert = get_abs_path(__file__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MaxPressure 信号控制')
    parser.add_argument('--junction', type=str, default='Beijing_Beihuan',
                        help='路口名称')
    parser.add_argument('--env_name', type=str, default='easy_low_density',
                        help='环境名称，如 easy_low_density')
    parser.add_argument('--use_gui', action='store_true', default=False,
                        help='是否开启 GUI')
    parser.add_argument('--min_green_steps', type=int, default=2,
                        help='单个相位最小连续绿灯决策步数')
    parser.add_argument('--max_green_steps', type=int, default=10,
                        help='单个相位最大连续绿灯决策步数')
    parser.add_argument('--event_name', type=str, default='',
                        help='特殊事件集合名称（定义在 junction_configs 的 EVENTS 中，如 event_1）；为空则不注入事件')
    args = parser.parse_args()

    cfg = load_junction_config(args.junction, args.env_name)

    trip_info, fcd_output = generate_output_paths(args.junction, args.env_name, "maxpressure")

    # 特殊事件配置（来自路口配置文件的 EVENTS 字典）
    accident_configs, special_vehicle_configs = (
        load_event_config(args.junction, args.event_name) if args.event_name else ([], [])
    )

    env = make_env(
        sumo_cfg=cfg['sumo_cfg'], net_file=cfg['net_file'],
        tls_id=cfg['tls_id'], num_seconds=cfg['num_seconds'],
        use_gui=args.use_gui,
        trip_info=trip_info,
        fcd_output=fcd_output,
        accident_configs=accident_configs,
        special_vehicle_configs=special_vehicle_configs,
    )
    agent = MaxPressureAgent(min_green_steps=args.min_green_steps, max_green_steps=args.max_green_steps)
    agent.run(env, num_episodes=1)
