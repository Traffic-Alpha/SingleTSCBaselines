'''
Author: WANG Maonan
Date: 2025-07-17 13:01:58
LastEditors: Please set LastEditors
Description: 车辆 Route 生成
LastEditTime: 2026-02-27 16:29:35
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
        '606446495#0.451': [9, 9, 9, 9, 9],
        '538625748.377': [10, 9, 10, 10, 10],
        '832511541#0.628': [9, 10, 9, 10, 10],
    },
    
    # 2. 波动通勤车流 (Fluctuating Commuter Flow)
    "fluctuating_commuter": {
        '606446495#0.451': [12, 17, 28, 17, 12],
        '538625748.377': [16, 16, 17, 17, 15],
        '832511541#0.628': [20, 18, 14, 10, 8]  # 早高峰
    },
    
    # 3. 饱和高密度车流 (Saturated High-Density Flow)
    "high_density": {
        '606446495#0.451': [20, 20, 21, 20, 21],
        '538625748.377': [18, 18, 18, 18, 18],
        '832511541#0.628': [20, 20, 20, 20, 20]
    },
    
    # 4. 随机扰动车流 (Random Perturbation Flow)
    "random_perturbation": {
        '606446495#0.451': [13, 13, 28, 12, 12],
        '538625748.377': [11, 11, 10, 10, 24],
        '832511541#0.628': [10, 22, 10, 10, 10]
    },
    
    # 5. 递增需求车流 (Increasing Demand Flow)
    "increasing_demand": {
        '606446495#0.451': [6, 8, 10, 18, 26],  # 后期加速
        '538625748.377': [8, 17, 19, 19, 20],  # 前期增长后趋稳
        '832511541#0.628': [10, 13, 18, 22, 25]  # 稳步增长
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
