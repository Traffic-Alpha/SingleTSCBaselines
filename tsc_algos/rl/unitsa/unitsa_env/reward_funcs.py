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
    max_cost: float = 300.0,
) -> float:
    """取当前决策间隔最后一帧的平均累计等待时间，并返回其负值。

    max_cost 对单步累计等待封顶（默认 300s，与 state 的 MAX_ACCUMULATED_WAITING_TIME
    一致）。正常/最坏基线策略单步 cost ≤ 151s，封顶只削策略崩溃时的病态长尾值
    （单步可冲到几百秒），避免极端 reward 撑爆 VecNormalize 的 running std、
    触发破坏性更新导致 policy collapse。设 max_cost<=0 关闭封顶。
    """
    if not tls_dynamic_features_seq:
        raise ValueError("UniTSA reward history frames must not be empty.")

    cost = movement_avg_accumulated_waiting_time(tls_dynamic_features_seq[-1])
    if max_cost and max_cost > 0:
        cost = min(cost, max_cost)
    return float(-cost)
