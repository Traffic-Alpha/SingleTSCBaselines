'''
Author: WANG Maonan
Date: 2026-04-14 21:21:35
@LastEditTime: 2026-06-02 20:40:57
@LastEditors: Please set LastEditors
Description: 仿真输出路径生成工具
'''
import os
from pathlib import Path

def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).resolve().parents[0]


def generate_output_paths(junction: str, env_name: str, algo_name: str) -> tuple:
    """生成带时间戳的 SUMO 输出文件路径

    Args:
        junction: 路口名称
        env_name: 环境名称
        algo_name: 算法名称

    Returns:
        (trip_info_path, fcd_output_path): tuple of output file paths
    """
    folder_name = f"{junction}_{env_name}"
    output_dir = os.path.join(get_project_root(), "results", algo_name, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, "trip_info.xml"), os.path.join(output_dir, "fcd_output.xml")
