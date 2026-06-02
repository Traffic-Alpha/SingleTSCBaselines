'''
@Author: WANG Maonan
@Date: 2023-09-01 13:45:26
@Description: 随机生成交通流
LastEditTime: 2026-01-10 17:59:30
'''
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from tshub.sumo_tools.generate_routes import generate_route

# 初始化日志
current_file_path = get_abs_path(__file__)
set_logger(current_file_path('./'), file_log_level='WARNING', terminal_log_level='INFO')

# 开启仿真 --> 指定 net 文件
sumo_net = current_file_path("./env/single_junction.net.xml")

# 指定要生成的路口 id 和探测器保存的位置
generate_route(
    sumo_net=sumo_net,
    interval=[3,3,3], # 每个间隔 3 分钟 
    edge_flow_per_minute={
        '-E1': [9, 12, 6],
        '-E2': [15, 7, 10],
        '-E3': [7, 10, 12],
        'E0': [10, 9, 12],
    }, # 每分钟每个 edge 有多少车
    edge_turndef={
        '-E2__-E0': [0.2, 0.1, 0.2],
        'E0__E3': [0.2, 0.2, 0.1],
        '-E3__E1': [0.1, 0.2, 0.3],
        '-E1__E2': [0.3, 0.3, 0.3],
    }, # 控制左传比例
    veh_type={
        'ego': {'color':'26, 188, 156', 'accel':1, 'decel':1, 'probability':0.1},
        'background': {'color':'155, 89, 182', 'speed':15, 'probability':0.9},
    },
    output_trip=current_file_path('./testflow.trip.xml'),
    output_turndef=current_file_path('./testflow.turndefs.xml'),
    output_route=current_file_path('./testflow.rou.xml'),
    interpolate_flow=False,
    interpolate_turndef=False,
)