'''
Author: WANG Maonan
Date: 2025-07-17 13:01:58
LastEditors: Please set LastEditors
Description: 车辆 Route 生成
LastEditTime: 2026-02-27 15:05:25
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
        '-588610201#2.87': [9, 9, 9, 9, 9],
        '588610204#0.116': [7, 8, 7, 7, 7],
        '588610201#0.932': [11, 12, 12, 12, 12],
        '-588610204#1.409': [8, 7, 8, 8, 8],
    },
    
    # 2. 波动通勤车流 (Fluctuating Commuter Flow)
    "fluctuating_commuter": {
        '-588610201#2.87': [6, 8, 11, 14, 16],  # 晚高峰
        '588610204#0.116': [6, 8, 13, 8, 6],  # 中段峰值
        '588610201#0.932': [10, 10, 11, 10, 9],  # 稳定支路
        '-588610204#1.409': [15, 13, 11, 8, 6],  # 早高峰
    },
    
    # 3. 饱和高密度车流 (Saturated High-Density Flow)
    "high_density": {
        '-588610201#2.87': [21, 21, 21, 21, 21],
        '588610204#0.116': [17, 17, 17, 17, 17],
        '588610201#0.932': [20, 20, 20, 20, 20],
        '-588610204#1.409': [19, 19, 19, 19, 19],
    },
    
    # 4. 随机扰动车流 (Random Perturbation Flow)
    "random_perturbation": {
        '-588610201#2.87': [28, 12, 12, 12, 12],
        '588610204#0.116': [12, 12, 26, 12, 11],
        '588610201#0.932': [12, 11, 11, 11, 26],
        '-588610204#1.409': [9, 20, 9, 9, 8],
    },
    
    # 5. 递增需求车流 (Increasing Demand Flow)
    "increasing_demand": {
        '-588610201#2.87': [11, 13, 15, 16, 18],  # 缓慢增长
        '588610204#0.116': [6, 7, 9, 17, 24],  # 后期加速
        '588610201#0.932': [7, 13, 15, 15, 16],  # 前期增长后趋稳
        '-588610204#1.409': [7, 10, 13, 16, 19],  # 稳步增长
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
