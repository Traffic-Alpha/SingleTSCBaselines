'''
@Author: WANG Maonan
@Description: Backward-compatible import shim for static feature tools.
'''
from .tools.static_features import (
    angle_to_vector,
    build_lane_phase_binding_mapping,
    build_lane_turn_function_mapping,
    calculate_normalized_length,
    calculate_normalized_position,
    extract_static_features,
)

__all__ = [
    'angle_to_vector',
    'build_lane_phase_binding_mapping',
    'build_lane_turn_function_mapping',
    'calculate_normalized_length',
    'calculate_normalized_position',
    'extract_static_features',
]
