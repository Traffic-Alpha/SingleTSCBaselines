'''
@Author: WANG Maonan
@Date: 2026-06-02 15:23:02
@Description: TLS/movement dynamic feature tools.
@LastEditTime: 2026-06-02 16:42:32
'''
SPEED_THRESHOLD = 1.0
MAX_LANE_NUMBER = 5.0


def get_index_value(values, index, default=0.0):
    """按 movement index 读取 tls 数组字段。"""
    if index < len(values):
        return values[index]
    return default


def direction_to_flags(direction):
    """将 movement 方向转换为 [straight, left, right]。"""
    direction = str(direction or '').lower()
    return [
        1 if direction in {'s', 'straight'} else 0,
        1 if direction in {'l', 'left'} else 0,
        1 if direction in {'r', 'right'} else 0,
    ]


def is_right_turn(direction):
    """判断 movement 是否为右转。"""
    return str(direction or '').lower() in {'r', 'right'}


def summarize_lanes(lane_ids, lane_dynamic_features):
    """汇总一组 lane 的动态统计。"""
    total_vehicles = 0
    waiting_vehicles = 0
    moving_vehicles = 0
    total_speed = 0.0
    total_waiting_time = 0.0
    total_accumulated_waiting_time = 0.0
    occupancy_values = []
    is_passable = False

    for lane_id in lane_ids:
        cells = lane_dynamic_features.get(lane_id, []) if lane_dynamic_features else []
        for cell in cells:
            vehicle_count = cell['vehicle_count']
            avg_speed = cell['avg_speed']

            total_vehicles += vehicle_count
            total_speed += avg_speed * vehicle_count
            total_waiting_time += cell['avg_waiting_time'] * vehicle_count
            total_accumulated_waiting_time += (
                cell['avg_accumulated_waiting_time'] * vehicle_count
            )
            occupancy_values.append(cell['occupancy'])
            if cell.get('is_passable', 0) > 0:
                is_passable = True

            if avg_speed < SPEED_THRESHOLD and vehicle_count > 0:
                waiting_vehicles += vehicle_count
            elif avg_speed >= SPEED_THRESHOLD and vehicle_count > 0:
                moving_vehicles += vehicle_count

    avg_speed = total_speed / total_vehicles if total_vehicles else 0.0
    avg_waiting_time = total_waiting_time / total_vehicles if total_vehicles else 0.0
    avg_accumulated_waiting_time = (
        total_accumulated_waiting_time / total_vehicles
        if total_vehicles else 0.0
    )
    avg_occupancy = (
        sum(occupancy_values) / len(occupancy_values)
        if occupancy_values else 0.0
    )

    return {
        'vehicle_count': float(total_vehicles),
        'waiting_vehicle_count': float(waiting_vehicles),
        'moving_vehicle_count': float(moving_vehicles),
        'avg_speed': float(avg_speed),
        'avg_waiting_time': float(avg_waiting_time),
        'avg_accumulated_waiting_time': float(avg_accumulated_waiting_time),
        'avg_occupancy': float(avg_occupancy),
        'is_passable': bool(is_passable),
    }


def build_movement_phase_indices(phase2movements):
    """构建 movement 到 phase index 列表的映射。"""
    movement_phase_indices = {}
    for phase_idx, movement_ids in phase2movements.items():
        for movement_id in movement_ids:
            movement_phase_indices.setdefault(movement_id, []).append(phase_idx)
    return movement_phase_indices


def extract_tls_dynamic_features(tls_state, lane_dynamic_features=None):
    """从原始 tls state 中提取信号灯动态特征，按 movement 组织。"""
    phase2movements = tls_state.get('phase2movements', {})
    movement_lane_ids = tls_state.get('movement_lane_ids', {})
    movement_ids = tls_state.get('movement_ids', list(movement_lane_ids.keys()))
    movement_directions = tls_state.get('movement_directions', {})
    movement_lane_numbers = tls_state.get('movement_lane_numbers', {})
    from_edge_to_edge = tls_state.get('fromEdge_toEdge', {})
    num_phases = len(phase2movements)
    this_phase_index = int(tls_state.get('this_phase_index', 0))
    current_movement_ids = set(phase2movements.get(this_phase_index, []))
    movement_phase_indices = build_movement_phase_indices(phase2movements)
    this_phase = tls_state.get('this_phase', [])
    last_phase = tls_state.get('last_phase', [])
    next_phase = tls_state.get('next_phase', [])
    jam_length_vehicle = tls_state.get('jam_length_vehicle', [])
    jam_length_meters = tls_state.get('jam_length_meters', [])
    last_step_occupancy = tls_state.get('last_step_occupancy', [])
    last_step_mean_speed = tls_state.get('last_step_mean_speed', [])
    last_step_vehicle_id_list = tls_state.get('last_step_vehicle_id_list', [])
    current_phase_onehot = [
        1.0 if phase_idx == this_phase_index else 0.0
        for phase_idx in range(num_phases)
    ]

    movement_features = []
    for movement_idx, movement_id in enumerate(movement_ids):
        incoming_lane_ids = movement_lane_ids.get(movement_id, [])
        edge_info = from_edge_to_edge.get(movement_id, [])
        from_edge = edge_info[0] if len(edge_info) > 0 else None
        to_edge = edge_info[1] if len(edge_info) > 1 else None
        to_lane = edge_info[3] if len(edge_info) > 3 else None
        outgoing_lane_ids = [to_lane] if to_lane else []

        incoming_summary = summarize_lanes(incoming_lane_ids, lane_dynamic_features)
        outgoing_summary = summarize_lanes(outgoing_lane_ids, lane_dynamic_features)
        direction = movement_directions.get(movement_id)
        lane_number = movement_lane_numbers.get(movement_id, len(incoming_lane_ids))
        occupancy = get_index_value(last_step_occupancy, movement_idx, 0.0)
        mean_speed = get_index_value(last_step_mean_speed, movement_idx, 0.0)
        phase_indices = movement_phase_indices.get(movement_id, []) # 这个 movement 受哪些 phase 控制
        right_turn = is_right_turn(direction)
        is_current_movement = bool(get_index_value(this_phase, movement_idx, False))
        is_current_movement = is_current_movement or movement_id in current_movement_ids
        is_passable = right_turn or is_current_movement or incoming_summary['is_passable']

        movement_features.append({
            'movement_index': movement_idx,
            'movement_id': movement_id,
            'this_phase_index': this_phase_index,
            'current_phase_onehot': current_phase_onehot,
            'can_perform_action': bool(tls_state.get('can_perform_action', False)),
            'this_phase': bool(get_index_value(this_phase, movement_idx, False)),
            'last_phase': bool(get_index_value(last_phase, movement_idx, False)),
            'next_phase': bool(get_index_value(next_phase, movement_idx, False)),
            'direction': direction,
            'direction_flags': direction_to_flags(direction),
            'phase_indices': phase_indices,
            'is_right_turn': right_turn,
            'is_controllable': bool(phase_indices),
            'is_current_phase': is_current_movement,
            'is_passable': bool(is_passable),
            'from_edge': from_edge,
            'to_edge': to_edge,
            'incoming_lane_ids': incoming_lane_ids,
            'outgoing_lane_ids': outgoing_lane_ids,
            'num_incoming_lanes': len(incoming_lane_ids),
            'num_outgoing_lanes': len(outgoing_lane_ids),
            'movement_lane_number': lane_number,
            'movement_lane_number_norm': lane_number / MAX_LANE_NUMBER,
            'vehicle_count': incoming_summary['vehicle_count'],
            'waiting_vehicle_count': incoming_summary['waiting_vehicle_count'],
            'moving_vehicle_count': incoming_summary['moving_vehicle_count'],
            'avg_speed': incoming_summary['avg_speed'],
            'avg_waiting_time': incoming_summary['avg_waiting_time'],
            'avg_accumulated_waiting_time': (
                incoming_summary['avg_accumulated_waiting_time']
            ),
            'avg_occupancy': incoming_summary['avg_occupancy'],
            'outgoing_vehicle_count': outgoing_summary['vehicle_count'],
            'outgoing_waiting_vehicle_count': (
                outgoing_summary['waiting_vehicle_count']
            ),
            'outgoing_moving_vehicle_count': outgoing_summary['moving_vehicle_count'],
            'outgoing_avg_speed': outgoing_summary['avg_speed'],
            'outgoing_avg_waiting_time': outgoing_summary['avg_waiting_time'],
            'outgoing_avg_accumulated_waiting_time': (
                outgoing_summary['avg_accumulated_waiting_time']
            ),
            'outgoing_avg_occupancy': outgoing_summary['avg_occupancy'],
            'pressure': (
                incoming_summary['waiting_vehicle_count']
                - outgoing_summary['waiting_vehicle_count']
            ),
            'jam_length_vehicle': get_index_value(jam_length_vehicle, movement_idx, 0.0),
            'jam_length_meters': get_index_value(jam_length_meters, movement_idx, 0.0),
            'last_step_occupancy': occupancy,
            'occupancy': occupancy / 100.0,
            'last_step_mean_speed': mean_speed,
            'mean_speed': mean_speed,
            'last_step_vehicle_id_list': get_index_value(
                last_step_vehicle_id_list,
                movement_idx,
                [],
            ),
        })

    return movement_features
