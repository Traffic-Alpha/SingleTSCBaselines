'''
Author: WANG Maonan
Date: 2025-07-17 13:01:58
LastEditors: WMN7 18811371255@163.com
Description: 车辆 Route 生成
LastEditTime: 2026-04-13 16:13:34
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
        '172801130.183': [6, 6, 6, 6, 6],
        '172801188#0.85': [7, 7, 7, 7, 7],
        '-172801188#1.174': [8, 7, 8, 7, 7],
    },
    
    # 2. 波动通勤车流 (Fluctuating Commuter Flow)
    "fluctuating_commuter": {
        '172801130.183': [20, 18, 14, 11, 8],  # 早高峰
        '172801188#0.85': [6, 9, 12, 14, 17],  # 晚高峰
        '-172801188#1.174': [9, 12, 19, 12, 8],  # 中段峰值
    },
    
    # 3. 饱和高密度车流 (Saturated High-Density Flow)
    "high_density": {
        '172801130.183': [15, 15, 15, 15, 15],
        '172801188#0.85': [16, 16, 16, 16, 16],
        '-172801188#1.174': [17, 17, 17, 17, 17],
    },
    
    # 4. 随机扰动车流 (Random Perturbation Flow)
    "random_perturbation": {
        '172801130.183': [11, 11, 11, 24, 11],
        '172801188#0.85': [25, 11, 11, 11, 11],
        '-172801188#1.174': [12, 12, 26, 12, 11],
    },
    
    # 5. 递增需求车流 (Increasing Demand Flow)
    "increasing_demand": {
        '172801130.183': [7, 9, 12, 15, 17],  # 稳步增长
        '172801188#0.85': [9, 11, 13, 14, 16],  # 缓慢增长
        '-172801188#1.174': [6, 7, 10, 18, 26],  # 后期加速
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
