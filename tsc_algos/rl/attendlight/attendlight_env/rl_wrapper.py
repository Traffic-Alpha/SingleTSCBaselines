'''
@Author: WANG Maonan
@Date: 2026-06-01 21:51:26
@Description: RL wrapper for AttendLight
@LastEditTime: 2026-06-02 14:16:25
'''
from typing import Any, Callable, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import Env


class ChooseNextPhaseWrapper(gym.Wrapper):
    """Convert TSCInfoWrapper output into the RL API for choose_next_phase."""

    def __init__(
        self,
        env: Env,
        reward_fn: Callable,
        state_fn: Callable,
        state_space: spaces.Space,
        num_phases: int,
        reward_scale: float = 1.0,
        state_kwargs: Dict[str, Any] = None,
        reward_kwargs: Dict[str, Any] = None,
    ) -> None:
        super().__init__(env)
        self.reward_fn = reward_fn
        self.state_fn = state_fn
        self.num_phases = num_phases
        self.reward_scale = reward_scale
        self.state_kwargs = state_kwargs or {}
        self.reward_kwargs = reward_kwargs or {}
        self._action_space = spaces.Discrete(num_phases)
        self._observation_space = state_space

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    def _get_tls_dynamic_features_seq(self, env_obs):
        return env_obs['tls_dynamic_features_seq']

    def _build_state(self, env_obs):
        """创建 movement-level RL state，并修正为固定 shape。"""
        tls_dynamic_features_seq = self._get_tls_dynamic_features_seq(env_obs)
        state = self.state_fn(
            tls_dynamic_features_seq,
            self.num_phases,
            **self.state_kwargs,
        ).astype(np.float32)

        target_shape = self._observation_space.shape
        if state.shape == target_shape:
            return state

        fixed_state = np.zeros(target_shape, dtype=np.float32)
        slices = tuple(
            slice(0, min(src_dim, dst_dim))
            for src_dim, dst_dim in zip(state.shape, target_shape)
        )
        fixed_state[slices] = state[slices]
        return fixed_state

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        env_obs, info = self.env.reset(seed=seed)
        return self._build_state(env_obs), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        env_obs, rewards, truncated, done, info = self.env.step(action)
        tls_dynamic_features_seq = self._get_tls_dynamic_features_seq(env_obs)
        reward = self.reward_fn(
            tls_dynamic_features_seq,
            **self.reward_kwargs,
        )
        return self._build_state(env_obs), reward * self.reward_scale, truncated, done, info
