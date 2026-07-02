'''
Author: WANG Maonan
Date: 2025-07-17 13:01:58
LastEditors: WMN7 18811371255@163.com
Description: 车辆 Route 生成
LastEditTime: 2026-04-13 16:36:48
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
        '417937574#1.74': [10, 10, 10, 10, 10],
        '417937497#0.2105': [10, 10, 10, 10, 10],
        '339537541#3': [10, 9, 10, 10, 9],
        '339537367#2.7': [9, 10, 9, 10, 10],
    },
    
    # 2. 波动通勤车流 (Fluctuating Commuter Flow)
    "fluctuating_commuter": {
        '339537541#3': [6, 9, 12, 14, 17],  # 晚高峰
        '339537367#2.7': [7, 10, 15, 9, 7],  # 中段峰值
        '417937574#1.74': [9, 9, 9, 9, 8],  # 稳定支路
        '417937497#0.2105': [21, 19, 15, 11, 8],  # 早高峰
    },
    
    # 3. 饱和高密度车流 (Saturated High-Density Flow)
    "high_density": {
        '339537541#3': [20, 19, 19, 20, 19],
        '339537367#2.7': [17, 18, 17, 17, 18],
        '417937574#1.74': [15, 16, 15, 16, 15],
        '417937497#0.2105': [18, 18, 19, 18, 19],
    },
    
    # 4. 随机扰动车流 (Random Perturbation Flow)
    "random_perturbation": {
        '339537541#3': [26, 12, 12, 12, 11],
        '339537367#2.7': [13, 13, 30, 13, 13],
        '417937574#1.74': [12, 25, 11, 11, 11],
        '417937497#0.2105': [13, 13, 13, 29, 12],
    },
    
    # 5. 递增需求车流 (Increasing Demand Flow)
    "increasing_demand": {
        '339537541#3': [9, 11, 13, 14, 16],  # 缓慢增长
        '339537367#2.7': [6, 8, 11, 19, 27],  # 后期加速
        '417937574#1.74': [6, 12, 14, 14, 14],  # 前期增长后趋稳
        '417937497#0.2105': [6, 9, 12, 15, 17],  # 稳步增长
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
