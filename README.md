<!--
 * @Author: WANG Maonan
 * @Date: 2026-01-10 17:05:12
 * @Description: 使用 TranssimHub 完成单路口信号灯控制的例子
 * @LastEditTime: 2026-04-21 11:47:50
-->
# 利用 TransSimHub 完成单智能体信号灯控制

## 项目结构

- `junction_scenarios/`: 单路口 SUMO 场景资产，包括路网、车流、检测器和 `.sumocfg` 文件。
- `junction_configs/`: 与场景对应的 Python 配置索引，包括信号灯 ID、相位数量、仿真时长和固定配时。
- `tsc_algos/`: 传统控制算法和强化学习算法实现。

## 基于 World Model 的决策

使用 diffusion transformer 来预测
