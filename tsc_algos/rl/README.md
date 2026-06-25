<!--
 * @Author: WANG Maonan
 * @Date: 2026-06-01 22:02:34
 * @Description: RL-based TSC Methods
 * @LastEditTime: 2026-06-25 21:17:32
-->
# RL Methods

这个目录保存 RL-based TSC 方法。当前重点维护：

- `presslight`: DQN + movement-level state + choose-next-phase action + pressure reward。
- `attendlight`: 独立环境代码复制自 `presslight` 的 state/action/reward 设计，只将特征提取器替换为 movement-token Transformer。
- `unitsa`: 和 `attendlight` 相比, 将代码替换为 PPO, action 使用 `next or not`, reward 使用 waiting time
- `intellilight`: DQN + movement-level state + choose-next-phase action + waiting time reward (vecnorm)。

### PressLight

### AttendLight

`attendlight` 使用独立的 `tsc_algos.rl.attendlight.attendlight_env.make_env`。这份环境代码从 PressLight 复制而来，因此 state、action 和 reward 设计保持一致，但后续可以单独演化：

- State: `(history_len, num_movements, movement_feature_dim)` 的 movement 历史序列。
- Action: `choose_next_phase`，动作空间为相位编号。
- Reward: 带历史窗口和时间衰减的 movement pressure reward。
- Model: movement row embedding + Transformer encoder + CLS token 输出。

### UniTSA

### 


训练：

```bash
python tsc_algos/rl/attendlight/train.py \
  --junction Beijing_Beihuan \
  --env_name normal_fluctuating_commuter \
  --num_envs 20 \
  --reward_scale 0.1 \
  --vec_env subproc \
  --history_len 5
```

评估：

```bash
python tsc_algos/rl/attendlight/eval.py \
  --junction Beijing_Beihuan \
  --env_name normal_fluctuating_commuter \
  --history_len 5
```
