'''
@Author: WANG Maonan
@Description: 可视化工具，集中所有 TSC 相关的可视化函数
+ 静态特征可视化（lane 位置和方向）
+ 动态拥堵热力图（单指标 / 多指标）
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow
from typing import Dict, List, Tuple, Any, Optional

from .tools import get_metric_value_range


def visualize_lane_features(
    static_lane_features: Dict[str, Dict[str, Any]],
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 12),
    arrow_scale: float = 0.3,
) -> None:
    """Visualize static features of lanes

    Plot each lane as an arrow starting from its position.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(0, 0, c='red', s=300, marker='X', zorder=5,
               edgecolors='darkred', linewidth=2, label='Junction Center')

    for lane_id, features in static_lane_features.items():
        position = features['position']
        heading = features['heading']
        x1, y1, x2, y2 = position
        vx, vy = heading

        ax.arrow(
            x1, y1, vx * arrow_scale, vy * arrow_scale,
            head_width=0.04, head_length=0.03,
            fc='green', ec='green', linewidth=1.5,
            zorder=3, alpha=0.8, length_includes_head=True
        )
        ax.scatter(x1, y1, c='blue', s=50, zorder=4, alpha=0.8,
                  edgecolors='black', linewidth=0.8)
        ax.scatter(x2, y2, c='blue', s=50, zorder=4, alpha=0.8,
                  edgecolors='black', linewidth=0.8)
        ax.plot([x1, x2], [y1, y2], c='grey', linewidth=1.5, zorder=2, alpha=0.3, linestyle='--')

    ax.set_xlabel('Relative X Coordinate (Normalized)', fontsize=12)
    ax.set_ylabel('Relative Y Coordinate (Normalized)', fontsize=12)
    ax.set_title('Lane Position and Direction Visualization', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

    legend_elements = [
        Line2D([0], [0], color='red', marker='X', linestyle='',
               markersize=12, markeredgewidth=2, markeredgecolor='darkred', label='Junction Center'),
        FancyArrow(0, 0, 0.1, 0, width=0.02, facecolor='blue', edgecolor='blue', label='Lane Position'),
        FancyArrow(0, 0, 0.1, 0, width=0.02, facecolor='green', edgecolor='green', label='Lane Direction'),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def visualize_lane_congestion(
    lane_dynamic_features: Dict[str, List[Dict]],
    lane_order: List[str] = None,
    metric: str = 'occupancy',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
    title: Optional[str] = None,
    cmap: str = None,
    show_values: bool = True,
    vmin: float = None,
    vmax: float = None
) -> None:
    """Visualize lane cell congestion heatmap"""
    if lane_order is None:
        lane_order = sorted(lane_dynamic_features.keys())

    max_cells = max(len(cells) for cells in lane_dynamic_features.values())
    num_lanes = len([lid for lid in lane_order if lid in lane_dynamic_features])
    data_matrix = np.full((num_lanes, max_cells), np.nan)

    lane_idx = 0
    for lane_id in lane_order:
        if lane_id not in lane_dynamic_features:
            continue
        cells = lane_dynamic_features[lane_id]
        for cell_idx, cell in enumerate(cells):
            if cell_idx >= max_cells:
                break
            data_matrix[lane_idx, cell_idx] = cell[metric]
        lane_idx += 1

    fig, ax = plt.subplots(figsize=figsize)

    if vmin is None or vmax is None:
        default_vmin, default_vmax = get_metric_value_range(metric)
        if vmin is None:
            vmin = default_vmin
        if vmax is None:
            vmax = default_vmax

    if cmap is None:
        default_cmaps = {
            'occupancy': 'YlOrRd', 'vehicle_count': 'Blues',
            'avg_speed': 'RdYlBu_r', 'avg_waiting_time': 'plasma',
            'avg_accumulated_waiting_time': 'viridis'
        }
        cmap_use = default_cmaps.get(metric, 'viridis')
    else:
        cmap_use = cmap

    masked_data = np.ma.masked_invalid(data_matrix)
    im = ax.imshow(masked_data, cmap=cmap_use, aspect='auto',
                   vmin=vmin, vmax=vmax, interpolation='nearest')

    metric_labels = {
        'occupancy': 'Occupancy', 'vehicle_count': 'Vehicle Count',
        'avg_speed': 'Average Speed (m/s)', 'avg_waiting_time': 'Average Waiting Time (s)',
        'avg_accumulated_waiting_time': 'Avg. Accumulated Waiting Time (s)'
    }
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_labels.get(metric, metric), fontsize=12)

    if show_values:
        for i in range(num_lanes):
            for j in range(max_cells):
                if not np.isnan(data_matrix[i, j]):
                    normalized_val = (data_matrix[i, j] - vmin) / (vmax - vmin + 1e-10)
                    text_color = 'white' if normalized_val > 0.5 else 'black'
                    text = f'{int(data_matrix[i, j])}' if metric == 'vehicle_count' else f'{data_matrix[i, j]:.2f}'
                    ax.text(j, i, text, ha='center', va='center', color=text_color, fontsize=8)

    valid_lane_ids = [lid for lid in lane_order if lid in lane_dynamic_features]
    ax.set_yticks(range(num_lanes))
    ax.set_yticklabels(valid_lane_ids, fontsize=9)
    ax.set_ylabel('Lane ID', fontsize=12, fontweight='bold')
    ax.set_xticks(range(max_cells))
    ax.set_xticklabels([f'Cell {i}' for i in range(max_cells)], rotation=45, ha='right')
    ax.set_xlabel('Cell Index', fontsize=12, fontweight='bold')

    if title is None:
        title = f'Lane Cell Congestion Heatmap - {metric_labels.get(metric, metric)}'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.set_xticks(np.arange(max_cells) - 0.5, minor=True)
    ax.set_yticks(np.arange(num_lanes) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def visualize_multiple_metrics(
    lane_dynamic_features: Dict[str, List[Dict]],
    lane_order: List[str] = None,
    metrics: List[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 12),
    title: Optional[str] = None
) -> None:
    """Visualize multiple metrics of lane cell congestion simultaneously"""
    if metrics is None:
        metrics = ['occupancy', 'vehicle_count', 'avg_speed', 'avg_waiting_time']
    if lane_order is None:
        lane_order = sorted(lane_dynamic_features.keys())

    max_cells = max(len(cells) for cells in lane_dynamic_features.values())
    num_lanes = len([lid for lid in lane_order if lid in lane_dynamic_features])

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    metric_labels = {
        'occupancy': 'Occupancy', 'vehicle_count': 'Vehicle Count',
        'avg_speed': 'Average Speed (m/s)', 'avg_waiting_time': 'Average Waiting Time (s)',
        'avg_accumulated_waiting_time': 'Avg. Accumulated Waiting Time (s)'
    }
    metric_cmaps = {
        'occupancy': 'YlOrRd', 'vehicle_count': 'Blues',
        'avg_speed': 'RdYlBu_r', 'avg_waiting_time': 'plasma',
        'avg_accumulated_waiting_time': 'viridis'
    }

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data_matrix = np.full((num_lanes, max_cells), np.nan)

        lane_idx = 0
        for lane_id in lane_order:
            if lane_id not in lane_dynamic_features:
                continue
            cells = lane_dynamic_features[lane_id]
            for cell_idx, cell in enumerate(cells):
                if cell_idx >= max_cells:
                    break
                data_matrix[lane_idx, cell_idx] = cell[metric]
            lane_idx += 1

        masked_data = np.ma.masked_invalid(data_matrix)
        im = ax.imshow(masked_data, cmap=metric_cmaps.get(metric, 'viridis'),
                       aspect='auto', interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_labels.get(metric, metric), fontsize=10)

        valid_lane_ids = [lid for lid in lane_order if lid in lane_dynamic_features]
        ax.set_yticks(range(num_lanes))
        ax.set_yticklabels(valid_lane_ids, fontsize=8)

        if idx == n_metrics - 1:
            ax.set_xticks(range(max_cells))
            ax.set_xticklabels([f'Cell {i}' for i in range(max_cells)], rotation=45, ha='right')
            ax.set_xlabel('Cell Index', fontsize=10, fontweight='bold')
        else:
            ax.set_xticks([])

        ax.set_ylabel('Lane ID', fontsize=10, fontweight='bold')
        ax.set_title(metric_labels.get(metric, metric), fontsize=11, fontweight='bold')
        ax.set_xticks(np.arange(max_cells) - 0.5, minor=True)
        ax.set_yticks(np.arange(num_lanes) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    if title is None:
        title = 'Lane Cell Multi-Metric Congestion Visualization'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
