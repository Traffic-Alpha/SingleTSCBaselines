'''
Author: WANG Maonan
Date: 2025-07-17 13:01:58
LastEditors: Please set LastEditors
Description: 车辆 Route 生成
LastEditTime: 2026-02-27 13:33:34
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
        '84355055#1': [6, 6, 6, 7, 6],
        '741602126#2.93': [7, 8, 7, 7, 8],
        '739536522.212': [8, 7, 8, 7, 7],
        '387284606#0.817': [7, 8, 7, 8, 7],
    },
    
    # 2. 波动通勤车流 (Fluctuating Commuter Flow)
    "fluctuating_commuter": {
        '84355055#1': [13, 14, 14, 14, 13],  # 稳定支路
        '741602126#2.93': [17, 14, 12, 9, 6],  # 早高峰
        '739536522.212': [5, 7, 9, 11, 13],  # 晚高峰
        '387284606#0.817': [8, 12, 18, 12, 8],  # 中段峰值
    },
    
    # 3. 饱和高密度车流 (Saturated High-Density Flow)
    "high_density": {
        '84355055#1': [18, 18, 18, 18, 18],
        '741602126#2.93': [17, 17, 17, 17, 17],
        '739536522.212': [17, 17, 17, 17, 17],
        '387284606#0.817': [18, 18, 18, 18, 18],
    },
    
    # 4. 随机扰动车流 (Random Perturbation Flow)
    "random_perturbation": {
        '84355055#1': [11, 11, 11, 25, 11],
        '741602126#2.93': [26, 12, 12, 12, 12],
        '739536522.212': [11, 11, 24, 11, 10],
        '387284606#0.817': [12, 11, 11, 11, 26],
    },
    
    # 5. 递增需求车流 (Increasing Demand Flow)
    "increasing_demand": {
        '84355055#1': [7, 13, 15, 16, 16],  # 前期增长后趋稳
        '741602126#2.93': [7, 10, 13, 17, 19],  # 稳步增长
        '739536522.212': [10, 12, 13, 14, 16],  # 缓慢增长
        '387284606#0.817': [7, 8, 11, 20, 29],  # 后期加速
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
