'''
@Author: WANG Maonan
@Description: 创建 SOTL 算法使用的环境
'''
from tsc_env import TSCEnvironment, TSCInfoWrapper, TSCEventWrapper


def make_env(sumo_cfg, net_file, tls_id, num_seconds, use_gui=False, cell_length=15.0, trip_info="", fcd_output="",
             accident_configs=None, special_vehicle_configs=None):
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
    if accident_configs or special_vehicle_configs:
        env = TSCEventWrapper(
            env,
            accident_configs=accident_configs,
            special_vehicle_configs=special_vehicle_configs,
        )
    env = TSCInfoWrapper(env, tls_id=tls_id, cell_length=cell_length)
    return env
