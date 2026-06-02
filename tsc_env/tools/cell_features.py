'''
@Author: WANG Maonan
@Description: Cell/lane dynamic feature tools based on vehicle states.
'''
from typing import Dict, List

import numpy as np


METRIC_VALUE_RANGES = {
    'occupancy': {
        'vmin': 0.0, 'vmax': 1.0,
        'description': 'Occupancy rate (0-1)', 'unit': ''
    },
    'vehicle_count': {
        'vmin': 0, 'vmax': 3,
        'description': 'Number of vehicles', 'unit': 'vehicles'
    },
    'avg_speed': {
        'vmin': 0.0, 'vmax': 15.0,
        'description': 'Average speed', 'unit': 'm/s'
    },
    'avg_waiting_time': {
        'vmin': 0.0, 'vmax': 120.0,
        'description': 'Average waiting time', 'unit': 's'
    },
    'avg_accumulated_waiting_time': {
        'vmin': 0.0, 'vmax': 300.0,
        'description': 'Average accumulated waiting time', 'unit': 's'
    },
    'distance_to_lane_start': {
        'vmin': 0.0, 'vmax': 100.0,
        'description': 'Distance to lane start', 'unit': 'm'
    },
    'is_passable': {
        'vmin': 0, 'vmax': 1,
        'description': 'Lane passability (0: red, 1: green)', 'unit': ''
    }
}


def get_metric_value_range(metric: str, custom_ranges: Dict[str, Dict] = None):
    """Get the recommended value range for a specific metric."""
    if custom_ranges and metric in custom_ranges:
        return custom_ranges[metric]['vmin'], custom_ranges[metric]['vmax']
    if metric in METRIC_VALUE_RANGES:
        return METRIC_VALUE_RANGES[metric]['vmin'], METRIC_VALUE_RANGES[metric]['vmax']
    return 0.0, 1.0


def update_metric_value_range(metric: str, vmin: float, vmax: float) -> None:
    """Update the global value range for a specific metric."""
    if metric in METRIC_VALUE_RANGES:
        METRIC_VALUE_RANGES[metric]['vmin'] = vmin
        METRIC_VALUE_RANGES[metric]['vmax'] = vmax
    else:
        METRIC_VALUE_RANGES[metric] = {
            'vmin': vmin, 'vmax': vmax,
            'description': metric, 'unit': ''
        }


def print_metric_value_ranges() -> None:
    """Print all metric value ranges for reference."""
    print("\n" + "=" * 70)
    print("Recommended Metric Value Ranges for Consistent Visualization")
    print("=" * 70)
    for metric, config in METRIC_VALUE_RANGES.items():
        print(f"\n{metric}:")
        print(f"  Range: [{config['vmin']}, {config['vmax']}] {config['unit']}")
        print(f"  Description: {config['description']}")
    print("=" * 70 + "\n")


class LaneCellManager:
    """管理 lane 的 cell 划分和动态信息计算。"""

    def __init__(
        self,
        static_lane_features: Dict[str, Dict],
        cell_length: float = 10.0,
    ) -> None:
        self.static_lane_features = static_lane_features
        self.cell_length = cell_length
        self.lane_cells_info = {}
        self._initialize_lane_cells()

    def _initialize_lane_cells(self) -> None:
        """初始化每条 lane 的 cell 划分信息。"""
        for lane_id, features in self.static_lane_features.items():
            lane_length = features['length'] * 100
            num_cells = int(np.ceil(lane_length / self.cell_length))

            io_type = features.get('io_type', [0, 0])
            is_incoming = (io_type[0] == 1)

            if is_incoming:
                cell_boundaries = []
                for i in range(num_cells + 1):
                    boundary = max(lane_length - i * self.cell_length, 0)
                    cell_boundaries.append(boundary)
                cell_boundaries = np.array(cell_boundaries)
            else:
                cell_boundaries = []
                for i in range(num_cells + 1):
                    boundary = min(i * self.cell_length, lane_length)
                    cell_boundaries.append(boundary)
                cell_boundaries = np.array(cell_boundaries)

            self.lane_cells_info[lane_id] = {
                'length': lane_length,
                'num_cells': num_cells,
                'cell_boundaries': cell_boundaries,
                'cell_centers': (cell_boundaries[:-1] + cell_boundaries[1:]) / 2,
                'is_incoming': is_incoming,
            }

    def __get_vehicle_cell_index(self, lane_id: str, lane_position: float) -> int:
        """获取车辆所在的 cell 索引。"""
        if lane_id not in self.lane_cells_info:
            return -1

        lane_info = self.lane_cells_info[lane_id]
        cell_boundaries = lane_info['cell_boundaries']
        num_cells = lane_info['num_cells']

        if lane_info['is_incoming']:
            cell_index = num_cells - np.searchsorted(
                cell_boundaries[::-1],
                lane_position,
                side='right',
            )
        else:
            cell_index = np.searchsorted(cell_boundaries, lane_position, side='right') - 1

        cell_index = np.clip(cell_index, 0, num_cells - 1)
        return int(cell_index)

    def calculate_lane_dynamic_features(
        self,
        vehicles_state: Dict[str, Dict],
        current_phase_index: int = None
    ) -> Dict[str, List[Dict]]:
        """计算每条 lane 中每个 cell 的动态特征。"""
        lane_cell_vehicles = {}
        for lane_id, lane_info in self.lane_cells_info.items():
            num_cells = lane_info['num_cells']
            lane_cell_vehicles[lane_id] = [[] for _ in range(num_cells)]

        for veh_id, veh_info in vehicles_state.items():
            lane_id = veh_info['lane_id']
            if lane_id not in self.lane_cells_info:
                continue
            lane_position = veh_info['lane_position']
            cell_index = self.__get_vehicle_cell_index(lane_id, lane_position)
            if cell_index >= 0:
                lane_cell_vehicles[lane_id][cell_index].append(veh_info)

        lane_dynamic_features = {}
        for lane_id, cells_vehicles in lane_cell_vehicles.items():
            lane_features = []
            cell_boundaries = self.lane_cells_info[lane_id]['cell_boundaries']
            is_passable = self._check_lane_passable(lane_id, current_phase_index)

            for cell_idx, vehicles in enumerate(cells_vehicles):
                is_incoming = self.lane_cells_info[lane_id]['is_incoming']
                if is_incoming:
                    cell_length = cell_boundaries[cell_idx] - cell_boundaries[cell_idx + 1]
                else:
                    cell_length = cell_boundaries[cell_idx + 1] - cell_boundaries[cell_idx]
                distance_to_lane_start = self.lane_cells_info[lane_id]['cell_centers'][cell_idx]

                if len(vehicles) == 0:
                    lane_features.append({
                        'vehicle_count': 0,
                        'avg_speed': 0.0,
                        'avg_waiting_time': 0.0,
                        'avg_accumulated_waiting_time': 0.0,
                        'distance_to_lane_start': distance_to_lane_start,
                        'occupancy': 0.0,
                        'is_passable': is_passable
                    })
                else:
                    speeds = [v['speed'] for v in vehicles]
                    waiting_times = [v['waiting_time'] for v in vehicles]
                    accumulated_waiting_times = [v['accumulated_waiting_time'] for v in vehicles]
                    vehicle_lengths = [v['length'] for v in vehicles]
                    total_vehicle_length = sum(vehicle_lengths)
                    occupancy = min(total_vehicle_length / cell_length, 1.0)

                    lane_features.append({
                        'vehicle_count': len(vehicles),
                        'avg_speed': np.mean(speeds),
                        'avg_waiting_time': np.mean(waiting_times),
                        'avg_accumulated_waiting_time': np.mean(accumulated_waiting_times),
                        'distance_to_lane_start': distance_to_lane_start,
                        'occupancy': occupancy,
                        'is_passable': is_passable
                    })

            lane_dynamic_features[lane_id] = lane_features

        return lane_dynamic_features

    def _check_lane_passable(self, lane_id: str, current_phase_index: int = None) -> int:
        """检查 lane 在当前相位是否可以通行。"""
        if current_phase_index is None:
            return 0
        if lane_id not in self.static_lane_features:
            return 0
        phase_binding = self.static_lane_features[lane_id].get('phase_binding', [])
        if not phase_binding or current_phase_index >= len(phase_binding):
            return 0
        return phase_binding[current_phase_index]

    def get_lane_summary(self, lane_dynamic_features: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """获取每条 lane 的汇总统计信息。"""
        lane_summary = {}
        for lane_id, cells in lane_dynamic_features.items():
            total_vehicles = sum(cell['vehicle_count'] for cell in cells)
            if total_vehicles == 0:
                lane_summary[lane_id] = {
                    'total_vehicles': 0,
                    'avg_speed': 0.0,
                    'avg_waiting_time': 0.0,
                    'avg_accumulated_waiting_time': 0.0,
                    'avg_occupancy': 0.0
                }
            else:
                weights = [cell['vehicle_count'] for cell in cells]
                total_weight = sum(weights)
                lane_summary[lane_id] = {
                    'total_vehicles': total_vehicles,
                    'avg_speed': sum(c['avg_speed'] * w for c, w in zip(cells, weights)) / total_weight,
                    'avg_waiting_time': sum(c['avg_waiting_time'] * w for c, w in zip(cells, weights)) / total_weight,
                    'avg_accumulated_waiting_time': sum(c['avg_accumulated_waiting_time'] * w for c, w in zip(cells, weights)) / total_weight,
                    'avg_occupancy': np.mean([c['occupancy'] for c in cells])
                }
        return lane_summary


def aggregate_features_seq(
    seq: List[Dict[str, List[Dict]]],
    method: str = 'last'
) -> Dict[str, List[Dict]]:
    """将特征序列聚合为单个快照。"""
    if method == 'last':
        return seq[-1]

    lane_ids = list(seq[0].keys())

    if method == 'mean':
        result = {}
        for lane_id in lane_ids:
            num_cells = len(seq[0][lane_id])
            cells_out = []
            for cell_idx in range(num_cells):
                keys = seq[0][lane_id][cell_idx].keys()
                agg = {}
                for k in keys:
                    vals = [step[lane_id][cell_idx][k] for step in seq]
                    agg[k] = float(np.mean(vals))
                cells_out.append(agg)
            result[lane_id] = cells_out
        return result

    if method == 'max':
        max_keys = {'vehicle_count', 'avg_waiting_time', 'avg_accumulated_waiting_time', 'occupancy'}
        min_keys = {'avg_speed'}
        result = {}
        for lane_id in lane_ids:
            num_cells = len(seq[0][lane_id])
            cells_out = []
            for cell_idx in range(num_cells):
                keys = seq[0][lane_id][cell_idx].keys()
                agg = {}
                for k in keys:
                    vals = [step[lane_id][cell_idx][k] for step in seq]
                    if k in max_keys:
                        agg[k] = float(np.max(vals))
                    elif k in min_keys:
                        agg[k] = float(np.min(vals))
                    else:
                        agg[k] = vals[-1]
                cells_out.append(agg)
            result[lane_id] = cells_out
        return result

    raise ValueError(f"Unknown aggregation method: {method!r}. Choose 'last', 'mean', or 'max'.")


def format_lane_features_to_array(
    lane_dynamic_features: Dict[str, List[Dict]],
    lane_order: List[str] = None,
    max_cells: int = None
) -> np.ndarray:
    """将 lane 动态特征格式化为数组。"""
    if lane_order is None:
        lane_order = sorted(lane_dynamic_features.keys())
    if max_cells is None:
        max_cells = max(len(cells) for cells in lane_dynamic_features.values())

    num_features = 6
    num_lanes = len([lid for lid in lane_order if lid in lane_dynamic_features])
    feature_array = np.zeros((num_lanes, max_cells, num_features), dtype=np.float32)

    lane_idx = 0
    for lane_id in lane_order:
        if lane_id not in lane_dynamic_features:
            continue
        cells = lane_dynamic_features[lane_id]
        for cell_idx, cell in enumerate(cells):
            if cell_idx >= max_cells:
                break
            feature_array[lane_idx, cell_idx, :] = [
                cell['vehicle_count'], cell['avg_speed'],
                cell['avg_waiting_time'], cell['avg_accumulated_waiting_time'],
                cell['occupancy'], cell.get('is_passable', 0)
            ]
        lane_idx += 1
    return feature_array


def create_lane_cell_mask(
    lane_cells_info: Dict[str, Dict],
    lane_order: List[str] = None,
    max_cells: int = None
) -> np.ndarray:
    """创建 lane cell 的 mask。"""
    if lane_order is None:
        lane_order = sorted(lane_cells_info.keys())
    if max_cells is None:
        max_cells = max(info['num_cells'] for info in lane_cells_info.values())

    num_lanes = len([lid for lid in lane_order if lid in lane_cells_info])
    mask = np.zeros((num_lanes, max_cells), dtype=bool)

    lane_idx = 0
    for lane_id in lane_order:
        if lane_id not in lane_cells_info:
            continue
        num_cells = lane_cells_info[lane_id]['num_cells']
        mask[lane_idx, :num_cells] = True
        lane_idx += 1
    return mask
