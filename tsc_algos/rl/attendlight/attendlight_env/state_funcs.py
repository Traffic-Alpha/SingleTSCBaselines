'''
@Author: WANG Maonan
@Date: 2026-06-02 15:59:09
@Description: AttendLight movement-based state functions
@LastEditTime: 2026-06-02 22:04:07
'''
from typing import Any, Dict, List

import numpy as np
from gymnasium import spaces


MAX_VEHICLES = 10.0
MAX_SPEED = 15.0
MOVEMENT_BASE_DIM = 9


def clip_norm(value: float, scale: float) -> float:
    """Normalize a non-negative feature to [0, 1]."""
    if scale <= 0:
        return 0.0
    return float(np.clip(value / scale, 0.0, 1.0))


def phase_multi_hot(phase_indices: List[int], num_phases: int) -> np.ndarray:
    """Build movement-to-phase binding vector."""
    phase_binding = np.zeros(num_phases, dtype=np.float32)
    for phase_idx in phase_indices:
        if 0 <= phase_idx < num_phases:
            phase_binding[phase_idx] = 1.0
    return phase_binding


def movement_frame_state(
    movement_features: List[Dict[str, Any]],
    num_phases: int,
) -> np.ndarray:
    """Build one AttendLight movement feature frame.

    Each row is one movement. phase_binding tells the model which action
    controls this movement; right-turn movements have all-zero phase_binding.
    """
    feature_dim = MOVEMENT_BASE_DIM + num_phases
    feature_array = np.zeros((len(movement_features), feature_dim), dtype=np.float32)

    for movement_idx, movement in enumerate(movement_features):
        direction_flags = movement['direction_flags']
        phase_binding = phase_multi_hot(movement['phase_indices'], num_phases)

        feature_array[movement_idx, 0] = clip_norm(
            len(movement['last_step_vehicle_id_list']),
            MAX_VEHICLES,
        )
        feature_array[movement_idx, 1] = clip_norm(
            movement['occupancy'],
            1.0,
        )
        feature_array[movement_idx, 2] = clip_norm(
            movement['mean_speed'],
            MAX_SPEED,
        )
        feature_array[movement_idx, 3] = clip_norm(
            movement['jam_length_vehicle'],
            MAX_VEHICLES,
        )
        feature_array[movement_idx, 4] = float(np.clip(
            movement['movement_lane_number_norm'],
            0.0,
            1.0,
        ))
        feature_array[movement_idx, 5] = float(movement['is_current_phase'])

        direction_cols = min(len(direction_flags), 3)
        feature_array[movement_idx, 6:6 + direction_cols] = direction_flags[:direction_cols]

        phase_start = MOVEMENT_BASE_DIM
        feature_array[movement_idx, phase_start:phase_start + num_phases] = phase_binding

    return feature_array


def movement_sequence_state(
    tls_dynamic_features_seq: List[List[Dict[str, Any]]],
    num_phases: int,
    history_len: int = 4,
) -> np.ndarray:
    """Build a fixed-length AttendLight time series state.

    The sequence is left-padded with all-zero frames at reset or when the
    current decision interval is shorter than history_len.
    """
    feature_dim = MOVEMENT_BASE_DIM + num_phases
    history_len = max(int(history_len), 1) # 时间序列
    frames = tls_dynamic_features_seq[-history_len:] # 取最近 history_len 帧的动态特征，较旧的帧会被丢弃
    frame_movement_counts = [len(frame) for frame in frames]
    if not frame_movement_counts:
        raise ValueError("AttendLight history frames must not be empty.")
    elif len(set(frame_movement_counts)) != 1:
        raise ValueError(
            "Movement count must be the same across AttendLight history frames, "
            f"got {frame_movement_counts}."
        )
    else:
        num_movements = frame_movement_counts[0]
    feature_array = np.zeros((history_len, num_movements, feature_dim), dtype=np.float32)

    start_idx = history_len - len(frames)
    for offset, movement_features in enumerate(frames):
        frame_state = movement_frame_state(movement_features, num_phases)
        expected_shape = (num_movements, feature_dim)
        if frame_state.shape != expected_shape:
            raise ValueError(
                f"Unexpected AttendLight frame shape {frame_state.shape}, "
                f"expected {expected_shape}."
            )
        feature_array[start_idx + offset] = frame_state

    return feature_array


def movement_sequence_state_space(
    num_phases: int,
    num_movements: int,
    history_len: int = 4,
) -> spaces.Box:
    """Return the observation space for movement_sequence_state."""
    return spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(max(int(history_len), 1), num_movements, MOVEMENT_BASE_DIM + num_phases),
        dtype=np.float32,
    )
