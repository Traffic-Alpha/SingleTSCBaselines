'''
Author: WANG Maonan
Date: 2025-07-17 13:01:58
LastEditors: WMN7 18811371255@163.com
Description: 车辆 Route 生成
LastEditTime: 2026-04-13 16:03:31
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
        '-885943869#2.897': [6, 7, 7, 7, 7],
        '170446483#0.356': [8, 7, 8, 8, 7],
        '885943869#0.263': [8, 8, 8, 8, 8],
        '806740830#0.118': [8, 7, 7, 7, 7],
    },
    
    # 2. 波动通勤车流 (Fluctuating Commuter Flow)
    "fluctuating_commuter": {
        '-885943869#2.897': [16, 16, 17, 17, 15],  # 稳定支路
        '806740830#0.118': [9, 13, 21, 13, 9],  # 中段峰值
        '170446483#0.356': [18, 15, 12, 9, 7],  # 早高峰
        '885943869#0.263': [7, 10, 13, 17, 19],  # 晚高峰
        
    },
    
    # 3. 饱和高密度车流 (Saturated High-Density Flow)
    "high_density": {
        '-885943869#2.897': [17, 17, 18, 17, 18],
        '806740830#0.118': [17, 17, 17, 16, 17],
        '170446483#0.356': [16, 16, 16, 17, 16],
        '885943869#0.263': [17, 18, 17, 17, 18],
    },
    
    # 4. 随机扰动车流 (Random Perturbation Flow)
    "random_perturbation": {
        '-885943869#2.897': [11, 11, 24, 10, 10],
        '806740830#0.118': [9, 9, 9, 20, 9],
        '170446483#0.356': [9, 9, 9, 9, 21],
        '885943869#0.263': [9, 20, 9, 9, 8],
    },
    
    # 5. 递增需求车流 (Increasing Demand Flow)
    "increasing_demand": {
        '-885943869#2.897': [7, 13, 15, 15, 15],  # 前期增长后趋稳
        '806740830#0.118': [6, 8, 10, 19, 27],  # 后期加速
        '170446483#0.356': [7, 10, 13, 17, 20],  # 稳步增长
        '885943869#0.263': [11, 13, 15, 16, 18],  # 缓慢增长
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
