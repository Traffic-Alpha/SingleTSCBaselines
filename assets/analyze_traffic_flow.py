"""
Traffic Flow Analysis Script
解析 junction_scenarios 中 12 个场景的 .rou.xml 文件，统计每分钟的车辆数
"""
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# 场景列表
SCENES = [
    "Beijing_Beihuan",
    "Beijing_Beishahe",
    "Beijing_Changjianglu",
    "Beijing_Gaojiaoyuan",
    "Beijing_Pinganli",
    "Beijing_Yongrunlu",
    "Chengdu_Chenghannanlu",
    "Chengdu_Guanghua",
    "France_Massy",
    "Hongkong_YMT",
    "SouthKorea_Songdo",
    "Tianjin_zhijingdao",
]

PATTERNS = [
    "low_density",
    "fluctuating_commuter",
    "high_density",
    "random_perturbation",
    "increasing_demand",
]

INTERVALS = 10  # 10 分钟，每分钟一个间隔
SIMULATION_TIME = 600  # 10 分钟 = 600 秒


def parse_rou_file(file_path: str) -> np.ndarray:
    """解析 rou.xml 文件，返回每分钟的车辆数"""
    counts = np.zeros(INTERVALS)

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        for vehicle in root.findall(".//vehicle"):
            depart = float(vehicle.get("depart", 0))
            if 0 <= depart < SIMULATION_TIME:
                interval_idx = int(depart // 60)
                if interval_idx >= INTERVALS:
                    interval_idx = INTERVALS - 1
                counts[interval_idx] += 1
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")

    return counts


def analyze_all_scenes(base_dir: str):
    """分析所有场景的数据"""
    all_data = {}  # scene -> pattern -> counts

    for scene in SCENES:
        all_data[scene] = {}
        scene_dir = os.path.join(base_dir, scene, "routes")

        for pattern in PATTERNS:
            rou_file = os.path.join(scene_dir, f"{pattern}.rou.xml")
            if os.path.exists(rou_file):
                all_data[scene][pattern] = parse_rou_file(rou_file)
            else:
                print(f"Warning: {rou_file} not found")
                all_data[scene][pattern] = np.zeros(INTERVALS)

    return all_data


def plot_traffic_demand_profiles(all_data: dict, output_dir: str):
    """绘制 4x3 网格的流量曲线图"""
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    colors = {
        "low_density": "#2ecc71",           # 绿色
        "fluctuating_commuter": "#3498db",  # 蓝色
        "high_density": "#e74c3c",          # 红色
        "random_perturbation": "#9b59b6",   # 紫色
        "increasing_demand": "#f39c12",      # 橙色
    }

    patterns_short = {
        "low_density": "Low",
        "fluctuating_commuter": "Fluct.",
        "high_density": "High",
        "random_perturbation": "Random",
        "increasing_demand": "Increas.",
    }

    x = np.arange(1, INTERVALS + 1)

    for idx, scene in enumerate(SCENES):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        for pattern in PATTERNS:
            counts = all_data[scene].get(pattern, np.zeros(INTERVALS))
            ax.plot(x, counts, color=colors[pattern],
                    marker='o', markersize=5, linewidth=2.0,
                    label=patterns_short[pattern])

        ax.set_title(scene.replace("_", " "), fontsize=13, fontweight='bold')
        ax.set_xlabel("Time (min)", fontsize=12)
        ax.set_ylabel("Vehicles/min", fontsize=12)
        ax.set_xlim(0.5, INTERVALS + 0.5)
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)

    # 在右上角添加共享图例
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=5,
              bbox_to_anchor=(0.98, 0.96), fontsize=12, framealpha=0.9)

    plt.savefig(os.path.join(output_dir, "traffic_demand_profiles.pdf"),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "traffic_demand_profiles.png"),
                dpi=300, bbox_inches='tight')

    plt.close()
    print(f"Figures saved to {output_dir}")


def generate_statistics_table(all_data: dict, output_dir: str):
    """生成统计表格 (LaTeX 格式)"""
    lines = []
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Traffic flow statistics for the 12 reconstructed scenes. "
                 "Values show mean $\\pm$ std vehicles per minute across the 10-minute simulation "
                 "(10 intervals of 1 minute each).}")
    lines.append("\\label{tab:traffic_flow_settings}")
    lines.append("\\begin{tabular}{lrrrrr}")
    lines.append("\\toprule")

    # 表头 - Mean ± Std 格式，std 用下标小字体
    pattern_short_names = {
        "low_density": "Low",
        "fluctuating_commuter": "Fluct.",
        "high_density": "High",
        "random_perturbation": "Random",
        "increasing_demand": "Incr.",
    }
    header_parts = ["\\textbf{Scene}"]
    for p in PATTERNS:
        header_parts.append(f"\\textbf{{{pattern_short_names[p]}}}")
    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    # 数据行 - Mean ± Std，std 下标小字体
    for scene in SCENES:
        row = [scene.replace("_", " ")]
        for pattern in PATTERNS:
            counts = all_data[scene].get(pattern, np.zeros(INTERVALS))
            mean_val = counts.mean()
            std_val = counts.std()
            row.append(f"${mean_val:.1f} \\pm {std_val:.1f}$")
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    table_path = os.path.join(output_dir, "traffic_flow_settings.tex")
    with open(table_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"Table saved to {table_path}")


def generate_csv_table(all_data: dict, output_dir: str):
    """生成 CSV 格式的统计表格（方便查看）"""
    import csv

    rows = [["Scene", "Pattern", "Total", "Mean", "Std"]]
    for scene in SCENES:
        for pattern in PATTERNS:
            counts = all_data[scene].get(pattern, np.zeros(INTERVALS))
            rows.append([
                scene,
                pattern,
                int(counts.sum()),
                f"{counts.mean():.1f}",
                f"{counts.std():.1f}",
            ])

    csv_path = os.path.join(output_dir, "traffic_flow_settings.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"CSV saved to {csv_path}")


def main():
    base_dir = "/home/wmn/Coding_Project/TrafficAlpha/SingleTSC_TranSimHub/junction_scenarios"
    output_dir = "/home/wmn/Coding_Project/TrafficAlpha/SingleTSC_TranSimHub/assets/figures/appendix_traffic_flow"

    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Analyzing traffic flow data...")
    all_data = analyze_all_scenes(base_dir)

    print("Generating figures...")
    plot_traffic_demand_profiles(all_data, output_dir)

    print("Generating tables...")
    table_output = "/home/wmn/Coding_Project/TrafficAlpha/SingleTSC_TranSimHub/assets/tables"
    Path(table_output).mkdir(parents=True, exist_ok=True)
    generate_statistics_table(all_data, table_output)
    generate_csv_table(all_data, table_output)

    print("Done!")


if __name__ == "__main__":
    main()