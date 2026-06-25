'''
@Author: WANG Maonan
@Description: UniTSA 环境组装
'''
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from tsc_env import TSCEnvironment, TSCInfoWrapper, TSCEventWrapper
from .reward_funcs import waiting_time_reward
from .rl_wrapper import NextOrNotWrapper
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
    reward_scale: float = 1.0,
    trip_info: str = "",
    fcd_output: str = "",
    accident_configs=None,
    special_vehicle_configs=None,
):
    """创建 UniTSA 环境

    Pipeline: TSCEnvironment -> [TSCEventWrapper] -> TSCInfoWrapper -> NextOrNotWrapper -> Monitor

    传入 accident_configs / special_vehicle_configs (来自 junction_configs 的 EVENTS) 时,
    在环境中注入特殊事件 (事故路障 / 特殊车辆), 用于评估鲁棒性。
    """
    def _init() -> gym.Env:
        env = TSCEnvironment(
            sumo_cfg=sumo_cfg,
            net_file=net_file,
            num_seconds=num_seconds,
            tls_ids=[tls_id],
            tls_action_type="next_or_not",
            use_gui=use_gui,
            trip_info=trip_info,
            fcd_output=fcd_output,
        )
        if accident_configs or special_vehicle_configs:
            env = TSCEventWrapper(
                env,
                accident_configs=accident_configs,
                special_vehicle_configs=special_vehicle_configs,
            )
        env = TSCInfoWrapper(env, tls_id=tls_id, cell_length=cell_length)
        env = NextOrNotWrapper(
            env,
            reward_fn=waiting_time_reward, # reward 使用所有车辆平均等待时间(取负)
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
            },
        )
        if log_file:
            env = Monitor(env, filename=f'{log_file}/{env_index}')
        return env

    return _init
