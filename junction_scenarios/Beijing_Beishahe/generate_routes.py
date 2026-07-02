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
        '741602130.216': [8, 8, 8, 8, 8],
        '657921289.337': [7, 7, 7, 7, 7],
        '657921284.293': [6, 6, 6, 6, 6]
    },
    
    # 2. 波动通勤车流 (Fluctuating Commuter Flow)
    "fluctuating_commuter": {
        '741602130.216': [8, 12, 15, 19, 22],  # 晚高峰
        '657921289.337': [10, 15, 24, 15, 10],  # 中段峰值
        '657921284.293': [11, 12, 12, 12, 11]  # 稳定支路
    },
    
    # 3. 饱和高密度车流 (Saturated High-Density Flow)
    "high_density": {
        '741602130.216': [25, 26, 25, 26, 25],
        '657921289.337': [22, 22, 22, 22, 22],
        '657921284.293': [20, 20, 20, 20, 20]
    },
    
    # 4. 随机扰动车流 (Random Perturbation Flow)
    "random_perturbation": {
        '741602130.216': [12, 26, 12, 12, 12],
        '657921289.337': [13, 13, 13, 29, 12],
        '657921284.293': [24, 11, 11, 10, 10]
    },
    
    # 5. 递增需求车流 (Increasing Demand Flow)
    "increasing_demand": {
        '741602130.216': [11, 14, 15, 16, 19],  # 缓慢增长
        '657921289.337': [7, 8, 11, 20, 29],  # 后期加速
        '657921284.293': [9, 18, 20, 20, 21]  # 前期增长后趋稳
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
