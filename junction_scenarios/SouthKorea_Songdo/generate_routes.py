'''
Author: WANG Maonan
Date: 2025-07-17 13:01:58
LastEditors: WMN7 18811371255@163.com
Description: 车辆 Route 生成
LastEditTime: 2026-04-13 16:24:15
'''
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.sumo_tools.generate_routes import generate_route

# 初始化日志
current_file_path = get_abs_path(__file__)
set_logger(current_file_path('./'), file_log_level='WARNING', terminal_log_level='INFO')

# 开启仿真 --> 指定 net 文件
sumo_net = current_file_path("./networks/easy.net.xml")

traffic_flow_configs = {
    # 1. 稳定低密度车流 (Stable Low-Density Flow)
    "low_density": {
        '989312046#0': [11, 11, 11, 11, 11],
        '315156946#0': [13, 13, 13, 13, 13],
        '525074961#6': [14, 14, 14, 14, 14],
        'E1': [12, 12, 12, 12, 12],
    },
    
    # 2. 波动通勤车流 (Fluctuating Commuter Flow)
    "fluctuating_commuter": {
        '989312046#0': [9, 12, 20, 12, 9],  # 中段峰值
        '315156946#0': [14, 15, 16, 16, 14],  # 稳定支路
        '525074961#6': [22, 19, 15, 12, 9],  # 早高峰
        'E1': [12, 16, 21, 27, 31],  # 晚高峰
    },
    
    # 3. 饱和高密度车流 (Saturated High-Density Flow)
    "high_density": {
        '989312046#0': [24, 24, 24, 24, 24],
        '315156946#0': [30, 30, 30, 30, 30],
        '525074961#6': [18, 18, 18, 18, 18],
        'E1': [23, 22, 23, 23, 23],
    },
    
    # 4. 随机扰动车流 (Random Perturbation Flow)
    "random_perturbation": {
        '989312046#0': [33, 15, 15, 15, 14],
        '315156946#0': [20, 20, 44, 19, 19],
        '525074961#6': [17, 17, 17, 17, 37],
        'E1': [19, 42, 19, 19, 18],
    },
    
    # 5. 递增需求车流 (Increasing Demand Flow)
    "increasing_demand": {
        '989312046#0': [9, 10, 14, 26, 36],  # 后期加速
        '315156946#0': [9, 18, 21, 21, 21],  # 前期增长后趋稳
        '525074961#6': [10, 13, 18, 22, 25],  # 稳步增长
        'E1': [14, 17, 19, 21, 24],  # 缓慢增长
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
