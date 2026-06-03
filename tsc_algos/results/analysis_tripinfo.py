'''
Author: WANG Maonan
Date: 2026-04-21 11:36:55
LastEditTime: 2026-04-21 12:53:33
LastEditors: WANG Maonan
Description: 分析 TripInfo 文件
'''
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.sumo_tools.analysis_output.tripinfo_analysis import TripInfoAnalysis

# 初始化日志
current_file_path = get_abs_path(__file__)
set_logger(current_file_path('./'), file_log_level="INFO")

tripinfos = [
    "./presslight/Beijing_Beihuan_normal_fluctuating_commuter/trip_info.xml",
    "./maxpressure/Beijing_Beihuan_normal_fluctuating_commuter/trip_info.xml",
]

for _tripinfo in tripinfos:
    tripinfo_file = current_file_path(_tripinfo)
    print(tripinfo_file)
    tripinfo_parser = TripInfoAnalysis(tripinfo_file)

    # 所有车辆一起分析
    stats = tripinfo_parser.calculate_multiple_stats(metrics=['duration', 'waitingTime'])
    TripInfoAnalysis.print_stats_as_table(stats)