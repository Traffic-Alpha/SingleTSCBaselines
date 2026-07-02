'''
@Author: WANG Maonan
@Description: UniTSA random-scenario training environment
'''
import random
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from junction_configs import load_junction_config

from .make_env import make_env


class RandomScenarioEnv(gym.Env):
    """Randomly select one env_name at reset and update SUMO inputs.

    This is intended for one junction and one network difficulty, e.g.
    SouthKorea_Songdo + easy + five traffic patterns. The observation/action
    contract must stay identical across candidate scenarios.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        junction: str,
        env_names: List[str],
        env_params: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not env_names:
            raise ValueError("RandomScenarioEnv requires at least one env_name.")

        self.junction = junction
        self.env_names = list(env_names)
        self.env_params = dict(env_params)
        self.scenario_cfgs = {
            env_name: load_junction_config(junction, env_name)
            for env_name in self.env_names
        }
        self.rng = random.Random(seed)
        self.current_env_name = self.env_names[0]
        self.current_env = self._build_env(self.current_env_name)

        self.action_space = self.current_env.action_space
        self.observation_space = self.current_env.observation_space

    def _build_env(self, env_name: str) -> gym.Env:
        cfg = self.scenario_cfgs[env_name]
        params = dict(self.env_params)
        params.update({
            'tls_id': cfg['tls_id'],
            'num_seconds': cfg['num_seconds'],
            'num_phases': cfg['num_phases'],
            'sumo_cfg': cfg['sumo_cfg'],
            'net_file': cfg['net_file'],
            'log_file': None,
        })
        return make_env(**params)()

    def _get_base_tsc_env(self):
        env = self.current_env
        while isinstance(env, gym.Wrapper):
            env = env.env
        if not hasattr(env, 'configure_sumo'):
            raise TypeError(
                "RandomScenarioEnv expects the base env to provide configure_sumo()."
            )
        return env

    def _configure_scenario(self, env_name: str) -> None:
        cfg = self.scenario_cfgs[env_name]
        self._get_base_tsc_env().configure_sumo(
            sumo_cfg=cfg['sumo_cfg'],
            net_file=cfg['net_file'],
            num_seconds=cfg['num_seconds'],
        )
        self.current_env_name = env_name

    def _close_current_env(self) -> None:
        if self.current_env is None:
            return
        try:
            self.current_env.close()
        except Exception:
            pass
        self.current_env = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)

        env_name = self.rng.choice(self.env_names)
        self._configure_scenario(env_name)

        obs, info = self.current_env.reset(seed=seed)
        info = dict(info)
        info['env_name'] = env_name
        return obs, info

    def step(self, action: int):
        if self.current_env is None:
            raise RuntimeError("RandomScenarioEnv.step() called before reset().")

        obs, reward, terminated, truncated, info = self.current_env.step(action)
        info = dict(info)
        info['env_name'] = self.current_env_name
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._close_current_env()


def make_random_scenario_env(
    junction: str,
    env_names: List[str],
    use_gui: bool = False,
    log_file: str = None,
    env_index: int = 0,
    cell_length: float = 15.0,
    num_movements: int = 12,
    history_len: int = 4,
    max_green: int = 45,
    reward_scale: float = 1.0,
    seed: Optional[int] = None,
    trip_info: str = "",
    fcd_output: str = "",
):
    """Create a UniTSA env that samples one scenario on each reset."""

    _validate_scenarios(junction, env_names)

    env_params = {
        'use_gui': use_gui,
        'env_index': env_index,
        'cell_length': cell_length,
        'num_movements': num_movements,
        'history_len': history_len,
        'max_green': max_green,
        'reward_scale': reward_scale,
        'trip_info': trip_info,
        'fcd_output': fcd_output,
    }

    def _init() -> gym.Env:
        env = RandomScenarioEnv(
            junction=junction,
            env_names=env_names,
            env_params=env_params,
            seed=seed,
        )
        if log_file:
            env = Monitor(
                env,
                filename=f'{log_file}/{env_index}',
                info_keywords=('env_name',),
            )
        return env

    return _init


def _validate_scenarios(junction: str, env_names: List[str]) -> None:
    """Check candidate scenarios share the same observation/action contract."""
    base_cfg = load_junction_config(junction, env_names[0])
    for env_name in env_names[1:]:
        cfg = load_junction_config(junction, env_name)
        if cfg['tls_id'] != base_cfg['tls_id']:
            raise ValueError(
                f"Scenario {env_name} has tls_id={cfg['tls_id']}, "
                f"expected {base_cfg['tls_id']}."
            )
        if cfg['num_phases'] != base_cfg['num_phases']:
            raise ValueError(
                f"Scenario {env_name} has num_phases={cfg['num_phases']}, "
                f"expected {base_cfg['num_phases']}."
            )
        if cfg['net_file'] != base_cfg['net_file']:
            raise ValueError(
                f"Scenario {env_name} uses a different net_file. "
                "RandomScenarioEnv should mix traffic patterns on one network."
            )
