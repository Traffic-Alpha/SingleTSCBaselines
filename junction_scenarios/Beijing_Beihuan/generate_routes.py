'''
Author: WANG Maonan
Date: 2025-07-17 13:01:58
LastEditors: Please set LastEditors
Description: 车辆 Route 生成
LastEditTime: 2026-02-27 12:50:36
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
        '157863208#0.1590': [10, 12, 9, 11, 10],
        '252712271#0.589': [8, 9, 7, 8, 8],
        '-1106233488.150': [10, 11, 9, 10, 10]
    },
    
    # 2. 波动通勤车流 (Fluctuating Commuter Flow)
    "fluctuating_commuter": {
        '157863208#0.1590': [10, 20, 10, 21, 17],  # 高峰低谷交替
        '252712271#0.589': [20, 14, 24, 11, 11],  # 强波动性
        '-1106233488.150': [15, 12, 13, 10, 14]   # 中等波动
    },
    
    # 3. 饱和高密度车流 (Saturated High-Density Flow)
    "high_density": {
        '157863208#0.1590': [21, 21, 15, 21, 15],
        '252712271#0.589': [15, 15, 20, 15, 15],
        '-1106233488.150': [19, 19, 16, 19, 17]
    },
    
    # 4. 随机扰动车流 (Random Perturbation Flow)
    "random_perturbation": {
        '157863208#0.1590': [10, 20, 16, 21, 16], 
        '252712271#0.589': [20, 10, 21, 9, 20], 
        '-1106233488.150': [10, 12, 20, 12, 11]
    },
    
    # 5. 递增需求车流 (Increasing Demand Flow)
    "increasing_demand": {
        '157863208#0.1590': [5, 10, 15, 20, 15],
        '252712271#0.589': [10, 10, 10, 15, 15],
        '-1106233488.150': [15, 10, 5, 20, 15]
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