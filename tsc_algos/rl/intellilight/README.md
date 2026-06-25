# IntelliLight

IntelliLight 是在 [PressLight](../presslight) 基础上的一个变体。**与 PressLight 相比只有两处不同**：

| 维度 | PressLight | IntelliLight |
|------|-----------|--------------|
| RL 算法 | DQN | DQN（不变） |
| 动作空间 | `choose_next_phase` | `choose_next_phase`（不变） |
| 状态 / 网络 | movement 序列 + MLP | movement 序列 + MLP（不变） |
| **Reward** | movement pressure | **所有车辆平均「累计」等待时间**（取负，与 UniTSA 一致） |
| **归一化** | 无 | **VecNormalize**（只归一化 reward，常开） |

> 用「累计」等待（`avg_accumulated_waiting_time`）而非瞬时等待，是为了抗 starvation：
> 被饿死的车其累计等待会无界增长，直接惩罚"放行大流量、饿死小流量方向"的策略。
> 累计等待 reward 尺度大且会随 starvation 增长，因此配 VecNormalize 稳住学习。

## 实验结果（Beijing_Beihuan / normal_fluctuating_commuter）

SUMO `tripinfo` 真实指标（300k 步训练，确定性策略评估）：

| 算法 | 平均行程 | 最大行程 | 平均等待 | 最大等待 |
|------|--------:|--------:|--------:|--------:|
| UniTSA（PPO + 累计等待） | 38.5 | 85 | **5.88** | **42** |
| AttendLight（DQN + pressure） | 38.6 | 114 | 5.94 | 82 |
| **IntelliLight（DQN + 累计等待）** | 39.2 | 101 | **6.27** | **56** |
| PressLight（DQN + pressure） | 40.9 | 149 | 7.78 | 85 |
| MaxPressure | 46.9 | 134 | 11.66 | 96 |

两组受控对照：

- **reward 的作用（IntelliLight vs PressLight，都是 DQN，只差 reward）**：平均等待
  6.27 < 7.78，最大等待 **56 ≪ 85** —— 累计等待 reward 明显优于 pressure，尤其大幅缓解
  starvation。**仅换 reward + VecNorm 就让 PressLight 提升一档。**
- **算法的作用（IntelliLight vs UniTSA，都是累计等待，只差 DQN↔PPO）**：平均等待
  6.27 vs 5.88，最大等待 56 vs 42 —— 同 reward 下 PPO 略优于 DQN。
- **关键规律**：两个累计等待方法（max 42 / 56）的最大等待远低于所有 pressure 方法
  （82–96），且**跨算法成立** —— 累计等待 reward 是抗 starvation 的决定性因素。

## 目录结构

```
intellilight/
  train.py                 # DQN 训练入口（VecNormalize 内置常开）
  eval.py                  # DQN 评估入口（--gui 可选；默认 headless 输出 tripinfo）
  model.py                 # IntelliLightMovementModel
  intellilight_env/
    make_env.py            # TSCEnvironment(choose_next_phase) -> TSCInfoWrapper -> ChooseNextPhaseWrapper -> Monitor
    rl_wrapper.py          # ChooseNextPhaseWrapper：动作空间 Discrete(num_phases)
    state_funcs.py         # movement-level 时序 state
    reward_funcs.py        # waiting_time_reward（累计等待，取负）
  checkpoints/             # 训练产物（best_model.zip / vec_normalize.pkl），已 gitignore
```

## 运行

训练（沙箱内用 `dummy`，本机可用 `subproc` 并行）：

```bash
conda run -n tshub python tsc_algos/rl/intellilight/train.py \
    --junction Beijing_Beihuan --env_name normal_fluctuating_commuter \
    --num_envs 20 --vec_env subproc --reward_scale 0.1 --history_len 5
```

评估（默认 headless，输出 SUMO `tripinfo` 到 `tsc_algos/results/intellilight/...`，
用 `--gui` 可开界面）：

```bash
conda run -n tshub python tsc_algos/rl/intellilight/eval.py \
    --junction Beijing_Beihuan --env_name normal_fluctuating_commuter --history_len 5
```

对比真实指标：用 `tshub` 的 `TripInfoAnalysis`（参考 `tsc_algos/results/analysis_tripinfo.py`）
读取各算法的 `trip_info.xml`，比较 `duration` / `waitingTime`。
