# AttendLight

Status: runnable approximation with an isolated AttendLight environment copy.

当前实现：

- Trainer: Stable-Baselines3 DQN。
- Env: 独立的 `attendlight_env`，state/action/reward 设计复制自 PressLight。
- Action: choose next phase。
- State: fixed-length movement time series，shape 为 `(history_len, num_movements, movement_feature_dim)`。
- Reward: weighted time-series movement pressure reward。
- Model: shared movement-row embedding + movement/time positional embedding + Transformer encoder + CLS token。

示例：

```bash
python tsc_algos/rl/attendlight/train.py \
  --junction Beijing_Beihuan \
  --env_name normal_fluctuating_commuter \
  --num_envs 20 \
  --reward_scale 0.1 \
  --vec_env subproc \
  --history_len 5

python tsc_algos/rl/attendlight/eval.py \
  --junction Beijing_Beihuan \
  --env_name normal_fluctuating_commuter \
  --history_len 5
```

后续可继续细化：

- 对齐原 AttendLight paper 中更细的 attention 结构。
- 做短程 sanity run 后调整 attention 层数、embedding 维度和 DQN 超参数。
