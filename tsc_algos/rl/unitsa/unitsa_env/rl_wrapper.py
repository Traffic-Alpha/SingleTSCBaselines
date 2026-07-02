'''
@Author: WANG Maonan
@Date: 2026-06-01 21:51:26
@Description: RL wrapper for UniTSA
@LastEditTime: 2026-07-01 17:36:39
'''
from typing import Any, Callable, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import Env


class NextOrNotWrapper(gym.Wrapper):
    """Convert TSCInfoWrapper output into the RL API for the next_or_not action.

    动作空间是 Discrete(2)：1=保持当前相位，0=切到下一相位（经黄灯）。num_phases 仅用于
    构建 movement-level state 的 phase_binding，与动作空间无关。
    """

    def __init__(
        self,
        env: Env,
        reward_fn: Callable,
        state_fn: Callable,
        state_space: spaces.Space,
        num_phases: int,
        reward_scale: float = 1.0,
        max_green: int = 45,
        state_kwargs: Dict[str, Any] = None,
        reward_kwargs: Dict[str, Any] = None,
    ) -> None:
        super().__init__(env)
        self.reward_fn = reward_fn
        self.state_fn = state_fn
        self.num_phases = num_phases
        self.reward_scale = reward_scale
        self.max_green = max(int(max_green), 0) if max_green is not None else 0
        self.state_kwargs = state_kwargs or {}
        self.reward_kwargs = reward_kwargs or {}
        self._action_space = spaces.Discrete(2)  # next_or_not: 0=切换, 1=保持
        self._observation_space = state_space
        self._current_phase_index = None
        self._current_green_time = 0
        self._last_step_time = 0

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    def _get_tls_dynamic_features_seq(self, env_obs):
        return env_obs['tls_dynamic_features_seq']

    def _get_current_phase_index(self, env_obs):
        tls_dynamic_features_seq = self._get_tls_dynamic_features_seq(env_obs)
        if not tls_dynamic_features_seq or not tls_dynamic_features_seq[-1]:
            return None
        return tls_dynamic_features_seq[-1][0].get('this_phase_index')

    def _to_discrete_action(self, action: int) -> int:
        return int(np.asarray(action).item())

    def _apply_max_green_guard(self, action: int) -> Tuple[int, bool]:
        """Force a phase change when the current green has been kept too long."""
        if self.max_green <= 0:
            return action, False
        if action == 1 and self._current_green_time >= self.max_green:
            return 0, True
        return action, False

    def _update_green_time(self, env_obs, info: Dict[str, Any]) -> None:
        phase_index = self._get_current_phase_index(env_obs)
        step_time = int(info.get('step_time', self._last_step_time))
        elapsed = max(step_time - self._last_step_time, 0)

        if self._current_phase_index is None or phase_index != self._current_phase_index:
            self._current_green_time = 0
        else:
            self._current_green_time += elapsed

        self._current_phase_index = phase_index
        self._last_step_time = step_time

    def _build_state(self, env_obs):
        """创建 movement-level RL state，并修正为固定 shape。"""
        tls_dynamic_features_seq = self._get_tls_dynamic_features_seq(env_obs)
        state_kwargs = dict(self.state_kwargs)
        state_kwargs.update({
            'current_green_time': self._current_green_time,
            'max_green': self.max_green,
        })
        state = self.state_fn(
            tls_dynamic_features_seq,
            self.num_phases,
            **state_kwargs,
        ).astype(np.float32)

        target_shape = self._observation_space.shape

        # 如果 state shape 与 observation_space shape 匹配，则直接返回
        if state.shape == target_shape:
            return state

        # 如果 state shape 不匹配，则修正为固定 shape（补零或截断）
        fixed_state = np.zeros(target_shape, dtype=np.float32)
        slices = tuple(
            slice(0, min(src_dim, dst_dim))
            for src_dim, dst_dim in zip(state.shape, target_shape)
        )
        fixed_state[slices] = state[slices]
        return fixed_state

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        env_obs, info = self.env.reset(seed=seed)
        self._current_phase_index = self._get_current_phase_index(env_obs)
        self._current_green_time = 0
        self._last_step_time = int(info.get('step_time', 0))
        return self._build_state(env_obs), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = self._to_discrete_action(action)
        action, max_green_forced = self._apply_max_green_guard(action)
        env_obs, rewards, truncated, done, info = self.env.step(action)
        previous_step_time = self._last_step_time
        self._update_green_time(env_obs, info)
        tls_dynamic_features_seq = self._get_tls_dynamic_features_seq(env_obs)
        reward = self.reward_fn(
            tls_dynamic_features_seq,
            **self.reward_kwargs,
        )
        info['applied_action'] = action
        info['max_green_forced'] = max_green_forced
        info['current_green_time'] = self._current_green_time
        info['decision_interval_seconds'] = max(
            int(info.get('step_time', previous_step_time)) - previous_step_time,
            0,
        )
        return self._build_state(env_obs), reward * self.reward_scale, truncated, done, info
