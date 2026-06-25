'''
@Author: WANG Maonan
@Description: UniTSA reward —— 所有车辆的平均「累计」等待时间（取负）
'''
from typing import Any, Dict, List


def movement_avg_accumulated_waiting_time(movement_features: List[Dict[str, Any]]) -> float:
    """所有进口车辆的平均累计等待时间（按车辆数加权各 movement 的 avg_accumulated_waiting_time）。

    用累计等待（而非瞬时等待）是为了抗 starvation：被饿死的车其累计等待会持续增长，直接
    惩罚"放行大流量、饿死小流量方向"的策略。
    注意：若多个 movement 共享进口车道，会有少量重复计数，但作为 RL reward 信号一致可用。
    """
    total_wait = 0.0
    total_vehicles = 0.0
    for movement in movement_features:
        count = movement['vehicle_count']
        total_wait += movement['avg_accumulated_waiting_time'] * count
        total_vehicles += count

    if total_vehicles <= 0:
        return 0.0  # 路口没有车时等待时间为 0
    return float(total_wait / total_vehicles)


def waiting_time_reward(
    tls_dynamic_features_seq: List[List[Dict[str, Any]]],
    history_len: int = None,
) -> float:
    """决策间隔内「所有车辆平均累计等待时间」的负值（对各子步等权平均）。"""
    if not tls_dynamic_features_seq:
        raise ValueError("UniTSA reward history frames must not be empty.")

    if history_len is None:
        frames = tls_dynamic_features_seq
    else:
        frames = tls_dynamic_features_seq[-max(int(history_len), 1):]

    waits = [movement_avg_accumulated_waiting_time(frame) for frame in frames]
    return float(-sum(waits) / len(waits))
