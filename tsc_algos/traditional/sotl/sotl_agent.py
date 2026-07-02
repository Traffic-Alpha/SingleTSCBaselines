'''
Author: WANG Maonan
Date: 2026-04-13 20:14:19
@LastEditTime: 2026-07-02 16:42:39
@LastEditors: WANG Maonan
Description: SOTL (Self-Organizing Traffic Light) 算法实现
  - 维护一个当前相位进口车道的等待车辆累积计数
  - 当累积计数超过阈值时，切换到下一个相位，否则保持当前相位
  - 当当前相位超过最大绿灯步数时，强制切换到下一个相位
'''
from typing import Dict, Any, List

from tsc_algos.traditional.base_traditional import BaseTraditionalAgent
from tsc_env.tools import aggregate_features_seq

class SOTLAgent(BaseTraditionalAgent):
    """SOTL 自组织信号灯控制算法

    维护当前相位的等待车辆累积计数，
    当累积计数超过阈值时切换到下一个相位。
    当当前相位持续绿灯步数超过 max_green_steps 时，强制切换，避免绿灯无法切换。

    Args:
        num_phases: 信号灯相位数量
        threshold: 切换相位的等待车辆累积阈值
        max_green_steps: 单个相位允许的最大连续绿灯决策步数，超过后强制切换
    """

    def __init__(self, num_phases: int = 4, threshold: int = 20, max_green_steps: int = 12):
        self.num_phases = num_phases
        self.threshold = threshold
        self.max_green_steps = max_green_steps
        self.current_phase = 0
        self.accumulated_count = 0  # 累积通过的车辆
        self.green_steps = 0        # 当前相位已持续的决策步数

    def choose_action(
        self,
        lane_dynamic_features_seq: List[Dict[str, Any]],
        static_lane_features: Dict[str, Any],
    ) -> int:
        """根据累积车辆服务量决定是否切换相位，并强制执行最大绿灯步数限制

        统计当前绿灯相位绑定的进口车道上靠近停止线（最后一个 cell）的车辆数，
        累加到计数器中（κ）。当 κ 超过阈值 θ 或绿灯步数超过 max_green_steps 时，
        切换到下一个相位。
        使用 aggregate_features_seq(method='max') 对序列聚合，取各指标最大值后统计一帧。

        Args:
            lane_dynamic_features_seq: 决策间隔内所有子步的特征序列
            static_lane_features: lane 静态特征

        Returns:
            action: 当前相位索引（保持或切换后）
        """
        # 用 max 聚合序列，取各指标在时间维度上的最大值
        lane_dynamic_features = aggregate_features_seq(lane_dynamic_features_seq, method='max')

        # 统计当前绿灯相位车道上驶近的车辆数（累积服务量）
        current_phase_vehicle_count = 0

        for lane_id, static_feat in static_lane_features.items():
            # 只考虑进口车道 (io_type=[1,0])
            if static_feat['io_type'][0] != 1:
                continue

            phase_binding = static_feat['phase_binding']
            # 检查该车道是否绑定到当前相位
            if self.current_phase < len(phase_binding) and phase_binding[self.current_phase] == 1:
                if lane_id in lane_dynamic_features:
                    cells = lane_dynamic_features[lane_id] # 获得 cell 的动态信息
                    if cells:
                        # 只统计最后一个 cell（靠近停止线的排队区）
                        current_phase_vehicle_count += cells[0]['vehicle_count']

        # 累加等待车辆数
        self.accumulated_count += current_phase_vehicle_count
        self.green_steps += 1

        # 判断是否需要切换相位：超过阈值或超过最大绿灯步数
        if self.accumulated_count >= self.threshold or self.green_steps >= self.max_green_steps:
            self.current_phase = (self.current_phase + 1) % self.num_phases
            self.accumulated_count = 0
            self.green_steps = 0

        return self.current_phase
