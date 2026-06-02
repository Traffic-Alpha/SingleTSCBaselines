'''
@Author: WANG Maonan
@Date: 2026-06-01 01:11:07
@Description: MaxPressure 最大压力算法
对每个 phase 计算 pressure = incoming_waiting - outgoing_waiting，选择 pressure 最大的 phase。
当当前相位未达到最小绿灯步数时，强制保持当前相位，确保车辆有足够时间通过路口。
当当前相位超过最大绿灯步数时，强制排除当前相位，避免绿灯长期无法切换。
@LastEditTime: 2026-06-02 17:40:33
'''
from typing import Dict, Any, List, Optional
from tsc_algos.traditional.base_traditional import BaseTraditionalAgent
from tsc_env.tools import aggregate_features_seq

# 速度阈值，低于此速度视为排队等待
SPEED_THRESHOLD = 1.0  # m/s

class MaxPressureAgent(BaseTraditionalAgent):
    """最大压力算法

    对每个 phase:
    1. 找到绑定到该 phase 的所有 incoming lane，统计排队车辆数
    2. 找到绑定到该 phase 的所有 outgoing lane，统计排队车辆数
    3. pressure = incoming_waiting - outgoing_waiting
    选择 pressure 最大的 phase。
    当当前相位持续绿灯步数未达到 min_green_steps 时，强制保持当前相位。
    当当前相位持续绿灯步数超过 max_green_steps 时，强制切换到其余相位中压力最大的一个。

    Args:
        min_green_steps: 单个相位必须持续的最小绿灯决策步数，未达到时强制保持
        max_green_steps: 单个相位允许的最大连续绿灯决策步数，超过后强制切换
    """

    def __init__(self, min_green_steps: int = 1, max_green_steps: int = 12):
        self.min_green_steps = min_green_steps
        self.max_green_steps = max_green_steps
        self.current_phase = 0
        self.green_steps = min_green_steps  # 初始化为 min_green_steps，允许第一步自由选择
        self._num_phases: Optional[int] = None  # 懒初始化，首次调用时从静态特征推断

    def _init(self, static_lane_features: Dict[str, Any]) -> None:
        """首次调用时推断相位数"""
        self._num_phases = max(
            (len(feat.get('phase_binding', [])) for feat in static_lane_features.values()),
            default=1,
        ) or 1

    def choose_action(
        self,
        lane_dynamic_features_seq: List[Dict[str, Any]],
        static_lane_features: Dict[str, Any],
    ) -> int:
        # 首次调用时推断 num_phases，避免每步重复遍历
        if self._num_phases is None:
            self._init(static_lane_features)
        num_phases = self._num_phases

        # 用 max 聚合序列，取决策间隔内各指标峰值，捕获间隔内最大排队压力
        lane_dynamic_features = aggregate_features_seq(lane_dynamic_features_seq, method='max')

        # 计算每个 lane 的排队车辆数
        lane_waiting = {}
        for lane_id, cells in lane_dynamic_features.items():
            waiting = sum(
                cell['vehicle_count']
                for cell in cells
                if cell['avg_speed'] < SPEED_THRESHOLD and cell['vehicle_count'] > 0
            )
            lane_waiting[lane_id] = waiting

        # 计算每个 phase 的 pressure
        phase_pressure = [0.0] * num_phases

        for phase_idx in range(num_phases):
            incoming_waiting = 0
            outgoing_waiting = 0

            for lane_id, features in static_lane_features.items():
                pb = features.get('phase_binding', [])
                if phase_idx >= len(pb) or pb[phase_idx] != 1:
                    continue

                io_type = features['io_type']
                w = lane_waiting.get(lane_id, 0)

                if io_type[0] == 1:  # Incoming
                    incoming_waiting += w
                elif io_type[1] == 1:  # Outgoing
                    outgoing_waiting += w

            phase_pressure[phase_idx] = incoming_waiting - outgoing_waiting

        # 未达到最小绿灯步数时，强制保持当前相位，确保车辆有足够时间通过路口
        if self.green_steps < self.min_green_steps:
            self.green_steps += 1
            return self.current_phase

        # 超过最大绿灯步数时，强制排除当前相位，避免绿灯无法切换
        if self.green_steps >= self.max_green_steps:
            candidates = [i for i in range(num_phases) if i != self.current_phase]
            if not candidates:
                candidates = [(self.current_phase + 1) % num_phases]
            best_phase = int(max(candidates, key=lambda i: phase_pressure[i]))
        else:
            best_phase = int(max(range(num_phases), key=lambda i: phase_pressure[i]))

        # 更新内部状态
        if best_phase == self.current_phase:
            self.green_steps += 1
        else:
            self.current_phase = best_phase
            self.green_steps = 1

        return best_phase
