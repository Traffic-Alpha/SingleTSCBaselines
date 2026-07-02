'''
Author: WANG Maonan
Date: 2025-07-17 13:01:58
LastEditors: WMN7 18811371255@163.com
Description: 车辆 Route 生成
LastEditTime: 2026-04-13 15:57:23
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
        '458749037#0.1098': [7, 7, 7, 7, 6],
        '471999771#0.1574': [6, 7, 7, 6, 7],
        '845412822.360': [7, 6, 7, 7, 7],
    },
    
    # 2. 波动通勤车流 (Fluctuating Commuter Flow)
    "fluctuating_commuter": {
        '458749037#0.1098': [8, 12, 18, 11, 8],  # 中段峰值
        '471999771#0.1574': [10, 10, 10, 10, 9],  # 稳定支路
        '845412822.360': [15, 13, 11, 8, 6],  # 早高峰
    },
    
    # 3. 饱和高密度车流 (Saturated High-Density Flow)
    "high_density": {
        '458749037#0.1098': [16, 16, 16, 16, 16],
        '471999771#0.1574': [19, 19, 19, 19, 19],
        '845412822.360': [14, 14, 14, 14, 14],
    },
    
    # 4. 随机扰动车流 (Random Perturbation Flow)
    "random_perturbation": {
        '458749037#0.1098': [8, 17, 8, 7, 7],
        '471999771#0.1574': [9, 8, 8, 19, 8],
        '845412822.360': [22, 10, 10, 10, 10],
    },
    
    # 5. 递增需求车流 (Increasing Demand Flow)
    "increasing_demand": {
        '458749037#0.1098': [6, 7, 9, 17, 23],  # 后期加速
        '471999771#0.1574': [5, 11, 13, 13, 13],  # 前期增长后趋稳
        '845412822.360': [6, 8, 10, 13, 15],  # 稳步增长
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
