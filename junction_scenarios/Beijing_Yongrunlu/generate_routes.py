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
        '-588610201#2.87': [10, 8, 11, 7, 9],
        '588610204#0.116': [8, 7, 6, 8, 7],
        '588610201#0.932': [12, 12, 13, 11, 11],
        '-588610204#1.409': [9, 8, 8, 6, 8],
    },
    
    # 2. 波动通勤车流 (Fluctuating Commuter Flow)
    "fluctuating_commuter": {
        '-588610201#2.87': [5, 24, 4, 18, 4],  # 高峰低谷交替
        '588610204#0.116': [4, 17, 5, 12, 3],  # 强波动性
        '588610201#0.932': [7, 18, 5, 14, 6],   # 中等波动
        '-588610204#1.409': [5, 15, 4, 21, 8], # 高峰低谷交替
    },
    
    # 3. 饱和高密度车流 (Saturated High-Density Flow)
    "high_density": {
        '-588610201#2.87': [21, 22, 20, 21, 21],
        '588610204#0.116': [17, 18, 16, 17, 17],
        '588610201#0.932': [20, 21, 19, 20, 20],
        '-588610204#1.409': [19, 20, 18, 19, 19],
    },
    
    # 4. 随机扰动车流 (Random Perturbation Flow)
    "random_perturbation": {
        '-588610201#2.87': [15, 13, 24, 15, 9], 
        '588610204#0.116': [14, 12, 22, 16, 9], 
        '588610201#0.932': [12, 10, 15, 19, 15],
        '-588610204#1.409': [9, 13, 17, 8, 8],
    },
    
    # 5. 递增需求车流 (Increasing Demand Flow)
    "increasing_demand": {
        '-588610201#2.87': [5, 10, 15, 19, 24],  # 线性增长
        '588610204#0.116': [8, 8, 15, 15, 17],  # 阶梯式增长
        '588610201#0.932': [6, 9, 13, 18, 20],  # 加速增长
        '-588610204#1.409': [7, 9, 12, 20, 17], # 加速增长
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