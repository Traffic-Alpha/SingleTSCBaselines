'''
@Author: Maonan Wang
@Date: 2025-01-15 16:53:53
@Description: 特殊事件 Wrapper

在仿真过程中注入「特殊事件」, 方便评估算法在异常场景下的鲁棒性:
  - accidents: 在指定时间/车道/位置放置一辆静止车辆作为路障 (停 duration 秒)
  - special_vehicles: 在指定时间发车的特殊车辆 (救护车/警车/消防车等), 沿指定 route 行驶

事件配置定义在各路口的 junction_configs/<junction>.py 的 EVENTS 字典中,
经 junction_configs.load_event_config 解析为 (accident_configs, special_vehicle_configs),
在 eval 的 make_env 中按需挂载本 Wrapper。

accident_configs 每项字段:        id, depart_time, edge_id, lane_index, position, type, duration
special_vehicle_configs 每项字段:  id, type, depart_time, route

管线位置:
  TSCEnvironment -> TSCEventWrapper -> TSCInfoWrapper -> ...

@LastEditors: Maonan Wang
@LastEditTime: 2026-06-25 18:21:09
'''
import gymnasium as gym
from loguru import logger
from typing import List, Dict, Any, Optional


class TSCEventWrapper(gym.Wrapper):
    """在仿真中注入事故路障与特殊车辆的 Wrapper。

    需要直接操作底层 SUMO 连接, 因此放在 TSCEnvironment 之上、TSCInfoWrapper 之下,
    这样既能拿到 sumo 连接, 又不会破坏上层 wrapper 看到的 (obs, reward, ...) 结构。
    """

    def __init__(
        self,
        env: gym.Env,
        accident_configs: Optional[List[Dict[str, Any]]] = None,
        special_vehicle_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(env)
        # 特殊事件的配置
        self.accident_configs = accident_configs or []
        self.special_vehicle_configs = special_vehicle_configs or []

        # 运行时状态
        self.conn = None                 # SUMO 连接, reset 后获取
        self.pending_vehicles = {}        # 待插入的路障车辆 {veh_id: info}
        self.inserted_vehicles = set()    # 已插入的路障车辆 id

    @property
    def _sumo(self):
        """底层 SUMO (libsumo / traci) 连接。"""
        # self.env 为 TSCEnvironment, 其内部持有 TshubEnvironment(self.tsc_env), 连接为 .sumo
        return self.env.tsc_env.sumo

    def reset(self, **kwargs):
        state_infos = self.env.reset(**kwargs)
        self.conn = self._sumo  # reset 后连接才建立, 此处重新获取

        # 清空上一轮的事件记录
        self.pending_vehicles = {}
        self.inserted_vehicles = set()

        # 事故路障: 仅登记, 等到 step 中到达 depart_time 再插入
        for accident in self.accident_configs:
            self._register_accident_vehicle(accident)

        # 特殊车辆: 直接 add, 交给 SUMO 在 depart_time 自动发车
        for vehicle in self.special_vehicle_configs:
            self._create_special_vehicle(vehicle)

        return state_infos

    def step(self, action):
        # 在环境 step 前, 插入已到达出发时间的路障车辆
        if self.conn is not None and self.pending_vehicles:
            current_time = self.conn.simulation.getTime()
            self._insert_pending_vehicles(current_time)

        return self.env.step(action)

    def close(self) -> None:
        return self.env.close()

    # ====== 特殊场景创建 =======
    def _register_accident_vehicle(self, accident_config: Dict[str, Any]) -> None:
        """登记事故路障车辆 (在 reset 时建立 route, step 到点后插入)。"""
        veh_id = accident_config["id"]
        edge_id = accident_config["edge_id"]
        veh_type = accident_config["type"]

        # 创建临时路线 (只含事故所在 edge)
        route_id = f"route_{veh_id}"
        self._ensure_route(route_id, [edge_id])
        self._ensure_vtype(veh_type)

        # 存储车辆信息, 等待 step 时插入
        self.pending_vehicles[veh_id] = {
            'route_id': route_id,
            'edge_id': edge_id,
            'lane_index': accident_config["lane_index"],
            'position': accident_config["position"],
            'duration': accident_config["duration"],
            'depart_time': accident_config["depart_time"],
            'veh_type': veh_type,
        }
        logger.info(
            f"SIM: 事故路障 {veh_id} 将在 {accident_config['depart_time']} 秒出现在 "
            f"{edge_id}-{accident_config['lane_index']} 的 {accident_config['position']} 米处"
        )

    def _insert_pending_vehicles(self, current_time: float) -> None:
        """插入所有到达出发时间且尚未插入的路障车辆。"""
        for veh_id, veh_info in list(self.pending_vehicles.items()):
            if veh_id in self.inserted_vehicles:
                continue
            if current_time < veh_info['depart_time']:
                continue

            lane_id = f"{veh_info['edge_id']}_{veh_info['lane_index']}"
            # 若目标位置附近已有车辆, 先移除, 避免插入冲突
            for existing_veh in self.conn.lane.getLastStepVehicleIDs(lane_id):
                existing_pos = self.conn.vehicle.getLanePosition(existing_veh)
                if abs(veh_info['position'] - existing_pos) < 1.0:
                    # 先取消订阅再移除, 否则 libsumo 在下一步会因悬空订阅报 "is not known"
                    try:
                        self.conn.vehicle.unsubscribe(existing_veh)
                    except Exception:
                        pass
                    self.conn.vehicle.remove(existing_veh)
                    logger.info(f"INFO: 删除车辆 {existing_veh} 以插入路障 {veh_id}")

            # 插入车辆并固定为路障
            self.conn.vehicle.add(
                vehID=veh_id,
                routeID=veh_info['route_id'],
                typeID=veh_info['veh_type'],
                depart="now",
                departLane=veh_info['lane_index'],
                departPos=veh_info['position'],
                departSpeed=0,
            )
            self.conn.vehicle.moveTo(
                vehID=veh_id,
                laneID=lane_id,
                pos=veh_info['position'],
            )
            self.conn.vehicle.setStop(
                vehID=veh_id,
                edgeID=veh_info['edge_id'],
                pos=veh_info['position'],
                laneIndex=veh_info['lane_index'],
                duration=veh_info['duration'],
            )
            self.inserted_vehicles.add(veh_id)
            logger.info(f"INFO: Time {current_time:.0f}, 成功插入路障 {veh_id}")

    def _create_special_vehicle(self, vehicle_config: Dict[str, Any]) -> None:
        """创建特殊车辆 (救护车/警车等), reset 时直接 add, 由 SUMO 在 depart_time 发车。"""
        veh_id = vehicle_config["id"]
        route_edges = vehicle_config["route"]
        vehicle_type = vehicle_config["type"]
        depart_time = vehicle_config["depart_time"]

        route_id = f"route_{veh_id}"
        self._ensure_route(route_id, route_edges)
        self._ensure_vtype(vehicle_type)

        self.conn.vehicle.add(
            vehID=veh_id,
            routeID=route_id,
            typeID=vehicle_type,
            depart=str(depart_time),  # 指定时间出发
            departLane=0,
            departPos=0,
            departSpeed=0,
        )
        logger.info(f"SIM: 特殊车辆 {veh_id} ({vehicle_type}) 将在 {depart_time} 秒出发")

    # ====== 工具函数 =======
    def _ensure_route(self, route_id: str, edges: List[str]) -> None:
        """幂等地添加 route, 已存在则跳过。"""
        try:
            if route_id in self.conn.route.getIDList():
                return
        except Exception:
            pass
        self.conn.route.add(route_id, edges)

    def _ensure_vtype(self, type_id: str) -> None:
        """确保车辆类型存在, 未定义则从默认类型复制一份, 避免插入时报错。"""
        try:
            if type_id in self.conn.vehicletype.getIDList():
                return
            self.conn.vehicletype.copy("DEFAULT_VEHTYPE", type_id)
            logger.warning(
                f"INFO: 车辆类型 '{type_id}' 未在 scenario 中定义, 已从 DEFAULT_VEHTYPE 复制创建"
            )
        except Exception as e:
            logger.warning(f"INFO: 检查/创建车辆类型 '{type_id}' 失败: {e}")
