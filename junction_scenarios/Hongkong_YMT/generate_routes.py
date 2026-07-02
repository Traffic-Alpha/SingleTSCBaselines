'''
Author: WANG Maonan
Date: 2025-07-17 13:01:58
LastEditors: WMN7 18811371255@163.com
Description: 车辆 Route 生成
LastEditTime: 2026-04-13 16:16:36
'''
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.sumo_tools.generate_routes import generate_route

# 初始化日志
current_file_path = get_abs_path(__file__)
set_logger(current_file_path('./'), file_log_level='WARNING', terminal_log_level='INFO')

# 开启仿真 --> 指定 net 文件
sumo_net = current_file_path("./networks/normal.net.xml")

traffic_flow_configs = {
    # 1. 稳定低密度车流 (Stable Low-Density Flow)
    "low_density": {
        '102454134#0': [6, 6, 6, 6, 7],
        '1200878753#0': [9, 8, 9, 8, 8],
        '30658263#0': [8, 8, 8, 8, 8],
        '960661806#0': [8, 8, 9, 8, 8],
    },
    
    # 2. 波动通勤车流 (Fluctuating Commuter Flow)
    "fluctuating_commuter": {
        '102454134#0': [5, 7, 9, 12, 13],  # 晚高峰
        '1200878753#0': [6, 9, 15, 9, 6],  # 中段峰值
        '30658263#0': [11, 11, 12, 12, 10],  # 稳定支路
        '960661806#0': [16, 14, 11, 8, 6],  # 早高峰
    },
    
    # 3. 饱和高密度车流 (Saturated High-Density Flow)
    "high_density": {
        '102454134#0': [18, 18, 17, 18, 18],
        '1200878753#0': [17, 16, 17, 16, 17],
        '30658263#0': [15, 15, 15, 15, 15],
        '960661806#0': [16, 17, 17, 17, 17],
    },
    
    # 4. 随机扰动车流 (Random Perturbation Flow)
    "random_perturbation": {
        '102454134#0': [8, 8, 8, 8, 18],
        '1200878753#0': [9, 19, 8, 8, 8],
        '30658263#0': [11, 10, 10, 23, 10],
        '960661806#0': [21, 9, 9, 9, 9],
    },
    
    # 5. 递增需求车流 (Increasing Demand Flow)
    "increasing_demand": {
        '102454134#0': [8, 9, 10, 12, 13],  # 缓慢增长
        '1200878753#0': [5, 6, 7, 13, 19],  # 后期加速
        '30658263#0': [6, 12, 13, 13, 14],  # 前期增长后趋稳
        '960661806#0': [7, 9, 13, 16, 18],  # 稳步增长
    },
}

for config_id, config_info in traffic_flow_configs.items():
    generate_route(
        sumo_net=sumo_net,
        interval=[2,2,2,2,2], # 共有 10 min
        edge_flow_per_minute=config_info,
        edge_turndef={},
        veh_type={
            'background': {'color':'220,220,220', 'length': 5, 'probability':1},
        },
        output_trip=current_file_path('./testflow.trip.xml'),
        output_turndef=current_file_path('./testflow.turndefs.xml'),
        output_route=current_file_path(f'./routes/{config_id}.rou.xml'),
        interpolate_flow=False,
        interpolate_turndef=False,
    )
