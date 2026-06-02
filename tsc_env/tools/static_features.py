'''
@Author: WANG Maonan
@Date: 2026-01-10
@Description: 静态特征工具函数
+ 计算归一化长度、位置、方向向量
+ 构建 lane 到转向功能和相位绑定的映射
+ 从环境 state 中提取所有 lane 的静态特征
LastEditTime: 2026-04-14 22:35:42
'''
import numpy as np
from typing import List, Tuple, Dict, Any

def angle_to_vector(angle: float, is_degrees: bool = False) -> List[float]:
    """将角度转换为单位方向向量

    Args:
        angle: 角度值（弧度或度数）
        is_degrees: 如果为 True，输入角度为度数；如果为 False，输入角度为弧度（默认）

    Returns:
        [vx, vy] 单位方向向量

    Notes:
        - 角度约定：0 度/弧度指向正东（正 X 轴方向）
        - 逆时针为正方向
    """
    if is_degrees:
        angle_rad = np.radians(angle)
    else:
        angle_rad = angle

    vx = np.cos(angle_rad)
    vy = np.sin(angle_rad)

    return [float(vx), float(vy)]


def calculate_normalized_length(length: float) -> float:
    """计算归一化的车道长度

    Args:
        length: 车道长度

    Returns:
        归一化后的长度 (除以 100)
    """
    return length / 100.0


def calculate_normalized_position(
    shape: List[Tuple[float, float]],
    junction_center: Tuple[float, float]
) -> List[float]:
    """计算车道相对于路口中心的归一化位置

    使用 shape 的最后一个点（车道出口位置），计算相对于路口中心的位置

    Args:
        shape: 车道的形状点列表 [(x1, y1), (x2, y2), ...]
        junction_center: 路口中心坐标 (x, y)

    Returns:
        [x1, y1, x2, y2] 归一化后的位置 (除以 100)
    """
    point1 = shape[0] # 起点
    point2 = shape[1] # 终点

    # 相对于路口中心的坐标
    rel_x1 = (point1[0] - junction_center[0]) / 100.0
    rel_y1 = (point1[1] - junction_center[1]) / 100.0

    rel_x2 = (point2[0] - junction_center[0]) / 100.0
    rel_y2 = (point2[1] - junction_center[1]) / 100.0

    return [rel_x1, rel_y1, rel_x2, rel_y2]


def build_lane_turn_function_mapping(
    movement_lanes: Dict[str, List[str]]
) -> Dict[str, List[int]]:
    """构建 lane 到转向功能的映射

    根据 movement_lanes 中的信息，判断每条车道的转向功能：
    - --s: Straight (直行) -> [1, 0, 0]
    - --l: Left (左转) -> [0, 1, 0]
    - --r: Right (右转) -> [0, 0, 1]
    - Other (Outgoing/Unknown) -> [0, 0, 0]

    Args:
        movement_lanes: movement 到 lane 列表的映射

    Returns:
        lane_to_turn: lane 到转向功能的映射
    """
    lane_to_turn = {}

    for movement_id, lanes in movement_lanes.items():
        if '--s' in movement_id:
            turn_function = [1, 0, 0]  # Straight
        elif '--l' in movement_id:
            turn_function = [0, 1, 0]  # Left
        elif '--r' in movement_id:
            turn_function = [0, 0, 1]  # Right
        else:
            turn_function = [0, 0, 0]  # Other/Unknown

        for lane_id in lanes:
            lane_to_turn[lane_id] = turn_function

    return lane_to_turn


def build_lane_phase_binding_mapping(
    movement_lanes: Dict[str, List[str]],
    phase2movements: Dict[int, List[str]]
) -> Dict[str, List[int]]:
    """构建 lane 到相位绑定的映射. 
    例如 3 相位, [1, 0, 0] 表示这个 lane 受到 phase-0 控制

    根据 phase2movements 中的信息，判断每条车道属于哪些相位：
    - Incoming Lane: Multi-hot 编码，如果属于某个相位则对应位为 1
    - Outgoing Lane: 全 0 (不受灯控)

    Args:
        movement_lanes: movement 到 lane 列表的映射
        phase2movements: 相位到 movement 列表的映射

    Returns:
        lane_to_phases: lane 到相位绑定的映射
    """
    num_phases = len(phase2movements)
    lane_to_phases = {}

    # 首先构建 movement 到 phases 的映射
    movement_to_phases = {}
    for phase_id, movements in phase2movements.items():
        for movement_id in movements:
            if movement_id not in movement_to_phases:
                movement_to_phases[movement_id] = []
            movement_to_phases[movement_id].append(phase_id)

    # 然后为每个 lane 构建相位绑定
    for movement_id, lanes in movement_lanes.items():
        phase_binding = [0] * num_phases

        if movement_id in movement_to_phases:
            for phase_id in movement_to_phases[movement_id]:
                phase_binding[phase_id] = 1

        for lane_id in lanes:
            lane_to_phases[lane_id] = phase_binding

    return lane_to_phases


def extract_static_features(state: Dict, tls_id: str) -> Dict[str, Dict]:
    """从环境 state 中提取所有 lane 的静态特征

    每条 lane 的特征包括:
    1. I/O Type (2 维): [1,0] Incoming, [0,1] Outgoing
    2. Turn Function (3 维): Straight/Left/Right
    3. Phase Binding (num_phases 维): Multi-hot
    4. Lane Length (1 维): 归一化
    5. Lane Position (4 维): 相对路口中心
    6. Heading Vector (2 维): 方向向量

    Args:
        state: 环境状态
        tls_id: 交通信号灯 ID

    Returns:
        lane_features: 每条 lane 的静态特征字典
    """
    # 获得路口中心坐标
    junction_center = state['node'][tls_id]['node_coord']
    # 获得 in roads
    in_roads = state['tls'][tls_id]['in_roads']
    in_roads_heading = state['tls'][tls_id]['in_roads_heading']
    # 获得 out roads
    out_roads = state['tls'][tls_id]['out_roads']
    out_roads_heading = state['tls'][tls_id]['out_roads_heading']
    # 获得每个 movement 包含的 lanes
    movement_lanes = state['tls'][tls_id]['movement_lane_ids']
    phase2movement = state['tls'][tls_id]['phase2movements']

    # 提前构建 lane 到转向功能和相位绑定的映射（提高效率）
    lane_to_turn = build_lane_turn_function_mapping(movement_lanes)
    lane_to_phases = build_lane_phase_binding_mapping(movement_lanes, phase2movement)

    # 存储每个 lane 的静态特征
    lane_features = {}

    for lane_id, lane_info in state['lane'].items():
        _road_id = lane_id.split('_')[0]  # 获得 lane 所属的 road

        # 去掉不属于控制信号灯周围的 lane
        if _road_id not in in_roads and _road_id not in out_roads:
            continue

        features = {}

        # 1. I/O Type
        if _road_id in in_roads:
            features['io_type'] = [1, 0]
        elif _road_id in out_roads:
            features['io_type'] = [0, 1]
        else:
            features['io_type'] = [0, 0]

        # 2. Turn Function
        features['turn_function'] = lane_to_turn.get(lane_id, [0, 0, 0])

        # 3. Phase Binding
        features['phase_binding'] = lane_to_phases.get(lane_id, [0] * len(phase2movement))

        # 4. Lane Length (归一化)
        features['length'] = calculate_normalized_length(lane_info['length'])

        # 5. Lane Position
        features['position'] = calculate_normalized_position(
            lane_info['shape'],
            junction_center
        )

        # 6. Heading Vector
        # 这里 90-angle 是因为角度系统不一样
        if _road_id in in_roads:
            angle = 90 - in_roads_heading[_road_id]
            features['heading'] = angle_to_vector(angle, is_degrees=True)
        elif _road_id in out_roads:
            angle = 90 - out_roads_heading[_road_id]
            features['heading'] = angle_to_vector(angle, is_degrees=True)
        else:
            features['heading'] = [0, 0]

        lane_features[lane_id] = features

    return lane_features
