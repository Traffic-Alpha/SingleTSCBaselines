'''
Author: WANG Maonan
Date: 2026-04-21 11:36:55
@LastEditTime: 2026-06-25 00:00:00
@LastEditors: WANG Maonan
Description: 分析不同 TSC 方法在不同场景下的 TripInfo 文件, 计算并对比各方法的效率指标
    - travel_time (duration): 车辆从出发到到达的总耗时
    - waiting_time (waitingTime): 车辆累计等待 (速度 < 0.1m/s) 的时间
    - time_loss (timeLoss): 因低于理想速度行驶造成的时间损失

    结果目录结构约定:
        results/<algorithm>/<scenario>/trip_info.xml

    用法:
        python analysis_tripinfo.py                 # 分析所有方法/场景
        python analysis_tripinfo.py --csv out.csv   # 同时导出汇总 CSV
'''
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.sumo_tools.analysis_output.tripinfo_analysis import TripInfoAnalysis

# 初始化日志
current_file_path = get_abs_path(__file__)
set_logger(current_file_path('./'), file_log_level="INFO")

# 需要统计的指标 (TripInfo 列名 -> 展示名称)
METRICS = {
    'duration': 'travel_time',     # 平均行程时间
    'waitingTime': 'waiting_time', # 平均等待时间
    'timeLoss': 'time_loss',       # 平均时间损失
}
RESULTS_DIR = Path(current_file_path('./'))


def discover_tripinfos(results_dir: Path):
    """自动扫描 results 目录, 返回 {scenario: {algorithm: trip_info_path}}
    目录结构: results/<algorithm>/<scenario>/trip_info.xml
    """
    scenarios = defaultdict(dict)
    for trip_info in sorted(results_dir.glob('*/*/trip_info.xml')):
        algorithm = trip_info.parent.parent.name
        scenario = trip_info.parent.name
        scenarios[scenario][algorithm] = trip_info
    return scenarios


def summarize_tripinfo(trip_info_path: Path):
    """计算单个 trip_info.xml 的各指标均值, 返回 {展示名称: mean}"""
    parser = TripInfoAnalysis(str(trip_info_path))
    if parser.df is None or parser.df.empty:
        return None

    stats = parser.calculate_multiple_stats(metrics=list(METRICS.keys()))
    summary = {'num_vehicles': len(parser.df)}
    for raw_metric, nice_name in METRICS.items():
        if stats and raw_metric in stats:
            summary[nice_name] = stats[raw_metric]['mean']
        else:
            summary[nice_name] = float('nan')
    return summary


def build_comparison_table(scenarios):
    """为每个场景构建方法对比表 (行: 方法, 列: 指标均值)"""
    tables = {}
    for scenario, algo_paths in sorted(scenarios.items()):
        rows = {}
        for algorithm, trip_info_path in sorted(algo_paths.items()):
            summary = summarize_tripinfo(trip_info_path)
            if summary is not None:
                rows[algorithm] = summary
        if rows:
            df = pd.DataFrame.from_dict(rows, orient='index')
            # 按平均行程时间排序 (越小越优)
            df = df.sort_values('travel_time')
            tables[scenario] = df
    return tables


def main():
    arg_parser = argparse.ArgumentParser(description='分析 TSC 方法的 TripInfo 效率指标')
    arg_parser.add_argument('--csv', type=str, default=None,
                            help='将汇总结果导出到指定 CSV 文件')
    args = arg_parser.parse_args()

    scenarios = discover_tripinfos(RESULTS_DIR)
    if not scenarios:
        print(f'未在 {RESULTS_DIR} 下找到任何 trip_info.xml 文件.')
        return

    tables = build_comparison_table(scenarios)

    all_records = []  # 用于导出长表 CSV
    for scenario, df in tables.items():
        print(f'\n{"=" * 80}')
        print(f'Scenario: {scenario}')
        print(f'{"=" * 80}')
        # 数值保留两位小数, 车辆数为整数
        show = df.copy()
        for col in METRICS.values():
            show[col] = show[col].map(lambda v: f'{v:.2f}')
        show['num_vehicles'] = show['num_vehicles'].astype(int)
        print(show.to_string())

        for algorithm, row in df.iterrows():
            record = {'scenario': scenario, 'algorithm': algorithm}
            record.update(row.to_dict())
            all_records.append(record)

    if args.csv:
        out_df = pd.DataFrame(all_records)
        out_path = Path(args.csv)
        out_df.to_csv(out_path, index=False, float_format='%.2f')
        print(f'\n汇总结果已导出至: {out_path.resolve()}')


if __name__ == '__main__':
    main()
