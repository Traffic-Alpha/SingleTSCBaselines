'''
Author: WANG Maonan
Date: 2025-07-17 13:01:58
LastEditors: Please set LastEditors
Description: 车辆 Route 生成
LastEditTime: 2026-02-27 16:24:47
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
        '-156465481#3.628.100': [7, 7, 7, 7, 7],
        '169221931#0.91': [10, 9, 10, 9, 10],
        '156465483#0.660': [7, 8, 7, 8, 7],
        '33610069#0.1159': [10, 10, 10, 10, 10],
    },
    
    # 2. 波动通勤车流 (Fluctuating Commuter Flow)
    "fluctuating_commuter": {
        '-156465481#3.628.100': [18, 15, 12, 9, 7],  # 早高峰
        '169221931#0.91': [9, 12, 17, 21, 24],  # 晚高峰
        '156465483#0.660': [9, 12, 20, 12, 9],
        '33610069#0.1159': [13, 13, 14, 14, 13],  # 稳定支路
    },
    
    # 3. 饱和高密度车流 (Saturated High-Density Flow)
    "high_density": {
        '-156465481#3.628.100': [18, 18, 18, 18, 18],
        '169221931#0.91': [25, 25, 25, 25, 25],
        '156465483#0.660': [15, 15, 15, 15, 15],
        '33610069#0.1159': [21, 21, 21, 21, 21],
    },
    
    # 4. 随机扰动车流 (Random Perturbation Flow)
    "random_perturbation": {
        '-156465481#3.628.100': [9, 9, 9, 8, 19],
        '169221931#0.91': [11, 24, 11, 10, 10],
        '156465483#0.660': [9, 9, 9, 19, 8],
        '33610069#0.1159': [23, 10, 10, 10, 10],
    },
    
    # 5. 递增需求车流 (Increasing Demand Flow)
    "increasing_demand": {
        '-156465481#3.628.100': [7, 9, 12, 15, 17],  # 稳步增长
        '169221931#0.91': [13, 16, 18, 20, 22],  # 缓慢增长
        '156465483#0.660': [6, 7, 10, 18, 25],  # 后期加速
        '33610069#0.1159': [7, 14, 16, 17, 17],  # 前期增长后趋稳
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
