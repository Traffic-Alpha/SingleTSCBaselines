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
        '-156465481#3.628.100': [8, 6, 9, 5, 7],
        '169221931#0.91': [10, 12, 8, 9, 9],
        '156465483#0.660': [6, 8, 8, 7, 8],
        '33610069#0.1159': [10, 11, 9, 10, 10],
    },
    
    # 2. 波动通勤车流 (Fluctuating Commuter Flow)
    "fluctuating_commuter": {
        '-156465481#3.628.100': [5, 19, 14, 15, 8],
        '169221931#0.91': [5, 22, 25, 22, 9],
        '156465483#0.660': [7, 13, 16, 12, 14], 
        '33610069#0.1159': [10, 19, 18, 13, 7],
    },
    
    # 3. 饱和高密度车流 (Saturated High-Density Flow)
    "high_density": {
        '-156465481#3.628.100': [18, 19, 17, 18, 18],
        '169221931#0.91': [25, 26, 24, 25, 25],
        '156465483#0.660': [15, 16, 14, 15, 15],
        '33610069#0.1159': [21, 22, 20, 21, 21],
    },
    
    # 4. 随机扰动车流 (Random Perturbation Flow)
    "random_perturbation": {
        '-156465481#3.628.100': [11, 3, 17, 15, 8], 
        '169221931#0.91': [18, 3, 25, 8, 12], 
        '156465483#0.660': [9, 8, 14, 15, 8],
        '33610069#0.1159': [10, 18, 15, 12, 8],
    },
    
    # 5. 递增需求车流 (Increasing Demand Flow)
    "increasing_demand": {
        '-156465481#3.628.100': [6, 9, 12, 15, 18],
        '169221931#0.91': [11, 15, 18, 25, 20],
        '156465483#0.660': [8, 10, 14, 19, 15],
        '33610069#0.1159': [8, 12, 16, 20, 15],
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