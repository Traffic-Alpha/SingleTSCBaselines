'''
@Author: WANG Maonan
@Description: TSC Info Wrapper
+ reset 时提取静态特征、初始化动态管理器
+ step 时收集完整子步序列，返回结构化 observation
@LastEditTime: 2026-06-02 15:39:39
'''
import gymnasium as gym
from gymnasium.core import Env
from typing import Any, SupportsFloat, Tuple, Dict

from .tools import LaneCellManager, extract_static_features, extract_tls_dynamic_features



class TSCInfoWrapper(gym.Wrapper):
    """TSC Info Wrapper - 从环境中提取静态和动态信息

    公开属性（外部可直接访问）:
    - static_lane_features: 车道静态特征字典
    - lane_dynamic_features_seq: 当前决策间隔内所有子步的特征序列
    - tls_dynamic_features_seq: 当前决策间隔内所有子步的信号灯动态特征序列
    - lane_cell_manager: LaneCellManager 实例
    - lane_order: lane 排序列表

    step() / reset() 均返回结构化 observation，包含 lane 和 tls 两类动态特征。
    """
    def __init__(self,
        env: Env,
        tls_id: str,
        cell_length: float = 15.0
    ) -> None:
        super().__init__(env)
        self.tls_id = tls_id
        self.cell_length = cell_length
        self.steps = 0

        self.static_lane_features = None
        self.lane_cell_manager = None
        self.lane_dynamic_features_seq = None
        self.tls_dynamic_features_seq = None
        self.lane_order = None

    def _build_lane_order(self):
        """确定 lane 排序: incoming lanes (sorted) + outgoing lanes (sorted)"""
        incoming = sorted(
            lid for lid, f in self.static_lane_features.items() if f['io_type'][0] == 1
        )
        outgoing = sorted(
            lid for lid, f in self.static_lane_features.items() if f['io_type'][1] == 1
        )
        return incoming + outgoing

    def _extract_lane_dynamic_features(self, state):
        """根据当前 state 计算一个子步的 cell-based 特征, 根据 cell 计算 lane 的特征"""
        phase_index_now = state['tls'][self.tls_id]['this_phase_index']
        return self.lane_cell_manager.calculate_lane_dynamic_features(
            vehicles_state=state['vehicle'],
            current_phase_index=phase_index_now
        )

    def _build_observation(self):
        """构建结构化 observation，让 lane/tls 动态特征处于同一层级。"""
        return {
            'lane_dynamic_features_seq': self.lane_dynamic_features_seq,
            'tls_dynamic_features_seq': self.tls_dynamic_features_seq,
        }

    def reset(self, seed=1) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """reset 时初始化静态信息和动态信息，返回长度为 1 的序列"""
        state = self.env.reset()
        self.steps = 0

        # 1. 提取静态特征（独立函数）
        self.static_lane_features = extract_static_features(state, self.tls_id)

        # 2. 初始化动态特征管理器
        self.lane_cell_manager = LaneCellManager(
            static_lane_features=self.static_lane_features,
            cell_length=self.cell_length,
        )

        # 3. 确定 lane 排序
        self.lane_order = self._build_lane_order()

        # 4. 计算初始 lane 动态特征，封装为序列
        initial_lane_features = self._extract_lane_dynamic_features(state)
        self.lane_dynamic_features_seq = [initial_lane_features]

        # 5. 计算初始 tls 动态特征，封装为序列
        initial_tls_features = extract_tls_dynamic_features(
            state['tls'][self.tls_id],
            initial_lane_features,
        )
        self.tls_dynamic_features_seq = [initial_tls_features]

        return self._build_observation(), {'step_time': 0}

    def step(self, action: int) -> Tuple[Dict[str, Any], SupportsFloat, bool, bool, Dict[str, Any]]:
        """执行一步环境交互，返回决策间隔内所有子步的特征序列"""
        can_perform_action = False
        lane_features_seq = []
        tls_features_seq = []
        while not can_perform_action:
            action_dict = {self.tls_id: action}
            states, rewards, truncated, dones, infos = super().step(action_dict)

            self.steps += 1
            can_perform_action = states['tls'][self.tls_id]['can_perform_action']

            # 提取 lane 的动态特征，封装为序列
            lane_features = self._extract_lane_dynamic_features(states)
            lane_features_seq.append(lane_features)

            # 提取 tls 的动态特征，封装为序列
            tls_features_seq.append(extract_tls_dynamic_features(
                states['tls'][self.tls_id],
                lane_features,
            ))

        self.lane_dynamic_features_seq = lane_features_seq
        self.tls_dynamic_features_seq = tls_features_seq

        infos['step_time'] = self.steps
        return self._build_observation(), rewards, truncated, dones, infos

    def close(self) -> None:
        return super().close()
