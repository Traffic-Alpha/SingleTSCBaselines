'''
@Author: WANG Maonan
@Date: 2026-06-02 16:39:14
@Description: PressLight reward functions
@LastEditTime: 2026-06-02 21:52:31
'''
from typing import Any, Dict, List


def movement_pressure_penalty(movement_features: List[Dict[str, Any]]) -> float:
    """Mean absolute pressure over phase-controlled movements."""
    pressure_penalty = 0.0
    controllable_count = 0
    for movement in movement_features:
        if not movement.get('is_controllable', bool(movement.get('phase_indices'))):
            continue
        pressure_penalty += abs(movement['pressure'])
        controllable_count += 1

    if controllable_count == 0:
        raise ValueError("PressLight reward requires at least one controllable movement.")

    return float(pressure_penalty / controllable_count)


def pressure_reward(
    tls_dynamic_features_seq: List[List[Dict[str, Any]]],
    history_len: int = None,
    time_decay: float = 1.0,
) -> float:
    """Time-series negative absolute pressure over phase-controlled movements.

    Right-turn and other always-passable movements have empty phase_indices
    and are ignored by this reward.

    The weighted mean keeps reward scale stable across movement counts and when
    the decision interval has more SUMO substeps. time_decay < 1.0 emphasizes
    recent frames.
    """
    if not tls_dynamic_features_seq:
        raise ValueError("PressLight reward history frames must not be empty.")

    if history_len is None:
        frames = tls_dynamic_features_seq
    else:
        frames = tls_dynamic_features_seq[-max(int(history_len), 1):]

    if not frames:
        raise ValueError("PressLight reward history frames must not be empty.")

    decay = float(time_decay)
    if decay < 0.0 or decay > 1.0:
        raise ValueError(f"reward time_decay must be in [0, 1], got {time_decay}.")

    weighted_penalty = 0.0
    weight_sum = 0.0
    for idx, movement_features in enumerate(frames):
        recency = len(frames) - idx - 1
        weight = decay ** recency if decay > 0 else float(recency == 0)
        weighted_penalty += movement_pressure_penalty(movement_features) * weight
        weight_sum += weight

    if weight_sum <= 0:
        return 0.0

    return float(-weighted_penalty / weight_sum)
