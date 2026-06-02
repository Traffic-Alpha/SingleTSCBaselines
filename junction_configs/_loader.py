'''
Author: WANG Maonan
Date: 2026-04-14 20:20:20
LastEditTime: 2026-04-14 20:28:24
LastEditors: WANG Maonan
Description: 路口配置加载器 - 根据路口名称和环境名称加载环境参数
    env_name 格式: {difficulty}_{scenario}，如 easy_low_density
'''
import os
import importlib

# 项目根目录（junction_configs 的上一级目录）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 所有可用路口名称
AVAILABLE_JUNCTIONS = [
    "Beijing_Beihuan",
    "Beijing_Beishahe",
    "Beijing_Changjianglu",
    "Beijing_Gaojiaoyuan",
    "Beijing_Pinganli",
    "Beijing_Yongrunlu",
    "Chengdu_Chenghannanlu",
    "Chengdu_Guanghua",
    "France_Massy",
    "Hongkong_YMT",
    "SouthKorea_Songdo",
    "Tianjin_zhijingdao",
]


def load_junction_config(junction_name, env_name):
    """加载路口配置，返回环境参数字典

    Args:
        junction_name: 路口名称，如 "Beijing_Beihuan"
        env_name: 环境名称，如 "easy_low_density"

    Returns:
        dict: {sumo_cfg, net_file, tls_id, num_phases, num_seconds, fix_phase_durations}
    """
    if junction_name not in AVAILABLE_JUNCTIONS:
        raise ValueError(
            f"未知路口 '{junction_name}'，可用路口: {AVAILABLE_JUNCTIONS}"
        )

    try:
        config_module = importlib.import_module(f"junction_configs.{junction_name}")
    except ModuleNotFoundError:
        raise ValueError(
            f"找不到路口配置文件 'junction_configs/{junction_name}.py'，"
            f"请确认该文件存在。"
        )

    junc_cfg = config_module.JUNCTION
    tls_id = junc_cfg["tls_id"]

    if env_name not in junc_cfg:
        available_envs = [k for k in junc_cfg.keys() if k != "tls_id"]
        raise ValueError(
            f"路口 '{junction_name}' 没有环境 '{env_name}'，"
            f"可用环境: {available_envs}"
        )

    env_cfg = junc_cfg[env_name]

    # 路径推导：env_name 格式为 "{difficulty}_{scenario}"
    # 例如 "easy_low_density" -> difficulty="easy"
    difficulty = env_name.split("_")[0]
    sumo_cfg = os.path.join(
        _PROJECT_ROOT, "exp_junction", junction_name,
        f"{env_name}.sumocfg"
    )
    net_file = os.path.join(
        _PROJECT_ROOT, "exp_junction", junction_name,
        "networks", f"{difficulty}.net.xml"
    )

    return {
        "sumo_cfg": sumo_cfg,
        "net_file": net_file,
        "tls_id": tls_id,
        "num_phases": env_cfg["num_phases"], # 相位数量
        "num_seconds": env_cfg["num_seconds"], # 仿真持续时间
        "fix_phase_durations": env_cfg.get("fix_phase_durations"), # 固定配时间设置
    }
