'''
@Author: WANG Maonan
@Date: 2023-09-04 20:43:53
@Description: 信号灯控制环境
LastEditTime: 2026-04-14 20:40:18
'''
import gymnasium as gym

from typing import Any, List, Dict, Optional
from tshub.tshub_env.tshub_env import TshubEnvironment

class TSCEnvironment(gym.Env):
    def __init__(self,
        sumo_cfg:str,
        net_file:str,
        num_seconds:int,
        tls_ids:List[str],
        tls_action_type:str="choose_next_phase",
        use_gui:bool=False,
        trip_info:str="",
        fcd_output:str="",
    ) -> None:
        super().__init__()

        self.tsc_env = TshubEnvironment(
            sumo_cfg=sumo_cfg,
            net_file=net_file,
            is_aircraft_builder_initialized=False,
            is_map_builder_initialized=True, # 获得地图信息
            is_vehicle_builder_initialized=True, # 获得车辆信息
            is_traffic_light_builder_initialized=True, # 信号灯相位信息
            tls_ids=tls_ids,
            num_seconds=num_seconds,
            tls_action_type=tls_action_type,
            use_gui=use_gui,
            is_libsumo=(not use_gui), # 如果不开界面, 就是用 libsumo
            # output: 空字符串视为不输出, 避免 SUMO 构建空路径输出文件报错
            trip_info=trip_info or None,
            fcd_output=fcd_output or None,
        )

    def reset(self) -> Dict[str, Any]:
        state_infos = self.tsc_env.reset()
        return state_infos

    def configure_sumo(
        self,
        sumo_cfg: str,
        net_file: Optional[str] = None,
        num_seconds: Optional[int] = None,
    ) -> None:
        """Update SUMO inputs used by the next reset."""
        self.tsc_env._sumo_cfg = sumo_cfg
        if net_file is not None:
            self.tsc_env._net = net_file
        if num_seconds is not None:
            self.tsc_env.sim_max_time = num_seconds

    def step(self, action:Dict[str, Dict[str, int]]):
        action = {'tls': action} # 这里只控制 tls 即可
        states, rewards, infos, dones = self.tsc_env.step(action)
        truncated = dones

        return states, rewards, truncated, dones, infos

    def close(self) -> None:
        self.tsc_env._close_simulation()
