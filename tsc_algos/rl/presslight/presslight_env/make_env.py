'''
@Author: WANG Maonan
@Date: 2026-06-01 01:11:07
@Description: 
@LastEditTime: 2026-06-02 15:50:48
'''
'''
@Author: WANG Maonan
@Description: PressLight 环境组装
'''
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from tsc_env import TSCEnvironment, TSCInfoWrapper
from .reward_funcs import pressure_reward
from .rl_wrapper import ChooseNextPhaseWrapper
from .state_funcs import movement_sequence_state, movement_sequence_state_space


def make_env(
    tls_id: str,
    num_seconds: int,
    num_phases: int,
    sumo_cfg: str,
    net_file: str,
    use_gui: bool = False,
    log_file: str = None,
    env_index: int = 0,
    cell_length: float = 15.0,
    num_movements: int = 12,
    history_len: int = 4,
    reward_time_decay: float = 1.0,
    reward_scale: float = 1.0,
    trip_info: str = "",
    fcd_output: str = "",
):
    """创建 PressLight 环境

    Pipeline: TSCEnvironment -> TSCInfoWrapper -> ChooseNextPhaseWrapper -> Monitor
    """
    def _init() -> gym.Env:
        env = TSCEnvironment(
            sumo_cfg=sumo_cfg,
            net_file=net_file,
            num_seconds=num_seconds,
            tls_ids=[tls_id],
            tls_action_type="choose_next_phase",
            use_gui=use_gui,
            trip_info=trip_info,
            fcd_output=fcd_output,
        )
        env = TSCInfoWrapper(env, tls_id=tls_id, cell_length=cell_length)
        env = ChooseNextPhaseWrapper(
            env,
            reward_fn=pressure_reward, # reward 使用 pressure 的设计
            state_fn=movement_sequence_state, # state 使用 movement-level tls 特征序列
            state_space=movement_sequence_state_space(
                num_phases=num_phases,
                num_movements=num_movements,
                history_len=history_len,
            ), # 定义 state space 的 shape 和范围
            num_phases=num_phases,
            reward_scale=reward_scale,
            state_kwargs={
                'history_len': history_len,
            },
            reward_kwargs={
                'history_len': history_len,
                'time_decay': reward_time_decay,
            },
        )
        if log_file:
            env = Monitor(env, filename=f'{log_file}/{env_index}')
        return env

    return _init
