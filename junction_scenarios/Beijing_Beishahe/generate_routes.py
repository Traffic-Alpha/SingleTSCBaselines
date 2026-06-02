'''
Author: WANG Maonan
Date: 2025-07-17 13:01:58
LastEditors: Please set LastEditors
Description: 车辆 Route 生成
LastEditTime: 2026-02-27 14:56:07
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
        '741602130.216': [8, 9, 7, 8, 8],
        '657921289.337': [7, 8, 6, 7, 7],
        '657921284.293': [6, 7, 5, 6, 6]
    },
    
    # 2. 波动通勤车流 (Fluctuating Commuter Flow)
    "fluctuating_commuter": {
        '741602130.216': [9, 20, 15, 18, 14],
        '657921289.337': [8, 22, 16, 15, 13],
        '657921284.293': [7, 18, 15, 12, 6]
    },
    
    # 3. 饱和高密度车流 (Saturated High-Density Flow)
    "high_density": {
        '741602130.216': [25, 27, 24, 26, 25],
        '657921289.337': [22, 23, 21, 22, 22],
        '657921284.293': [20, 21, 19, 20, 20]
    },
    
    # 4. 随机扰动车流 (Random Perturbation Flow)
    "random_perturbation": {
        '741602130.216': [15, 9, 25, 15, 10], 
        '657921289.337': [18, 12, 30, 8, 12], 
        '657921284.293': [12, 10, 9, 20, 15]
    },
    
    # 5. 递增需求车流 (Increasing Demand Flow)
    "increasing_demand": {
        '741602130.216': [5, 10, 15, 20, 25],
        '657921289.337': [8, 8, 15, 22, 22],
        '657921284.293': [10, 15, 25, 20, 18]
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