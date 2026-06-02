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
        '458749037#0.1098': [6, 7, 8, 6, 7],
        '471999771#0.1574': [7, 5, 8, 7, 6],
        '845412822.360': [8, 6, 7, 6, 7],
    },
    
    # 2. 波动通勤车流 (Fluctuating Commuter Flow)
    "fluctuating_commuter": {
        '458749037#0.1098': [8, 17, 12, 15, 5],  # 高峰低谷交替
        '471999771#0.1574': [4, 15, 13, 12, 5],  # 强波动性
        '845412822.360': [12, 8, 20, 7, 6],   # 中等波动
    },
    
    # 3. 饱和高密度车流 (Saturated High-Density Flow)
    "high_density": {
        '458749037#0.1098': [16, 16, 16, 16, 16],  # 持续高压
        '471999771#0.1574': [19, 19, 19, 19, 19],  # 稳定高负载
        '845412822.360': [13, 13, 15, 16, 13],   # 车道近饱和
    },
    
    # 4. 随机扰动车流 (Random Perturbation Flow)
    "random_perturbation": {
        '458749037#0.1098': [11, 6, 16, 8, 6], 
        '471999771#0.1574': [13, 5, 19, 7, 8], 
        '845412822.360': [15, 17, 8, 13, 9],
    },
    
    # 5. 递增需求车流 (Increasing Demand Flow)
    "increasing_demand": {
        '458749037#0.1098': [5, 7, 13, 18, 19],  # 加速增长 
        '471999771#0.1574': [5, 8, 10, 16, 16],  # 阶梯式增长
        '845412822.360': [4, 7, 9, 13, 19],  # 线性增长
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