# RL Methods

这个目录按照 RL-based TSC 的方法名组织，而不是按照 PPO、DQN 这类
训练器名称组织。一个方法目录对应 benchmark 表格里的一行方法。

每个方法由下面几个部分共同定义：

- action design
- state design
- reward design
- model design
- training / evaluation settings

每个方法目录应尽量自包含；环境接口、状态函数、奖励函数等和环境强相关
的代码放在该方法目录下的 `env/` 中。通用训练工具放在 `utils/`。

## Layout

```text
rl/
  README.md
  utils/
  presslight/
    env/
    models/
    train.py
    eval.py
  intellilight/
  frap/
  mplight/
  attendlight/
```

## Methods

| Method | Core idea | Action | State / model focus | Reward focus | Status |
| --- | --- | --- | --- | --- | --- |
| `presslight` | 将 max-pressure 思想引入 RL，学习 pressure-based 控制策略。 | 选择下一个相位 | lane aggregate state + MLP Q-network | pressure | runnable approximation |
| `intellilight` | 早期 DQN 风格 TSC 方法，强调可解释的交通状态和动作选择。 | 选择下一个相位 | lane traffic state + current green / phase binding | queue length | runnable approximation |
| `frap` | 建模 phase competition，让冲突相位根据 traffic demand 竞争。 | 选择下一个相位 | lane state + MLP placeholder | queue length | scaffold, phase competition pending |
| `mplight` | pressure-based scalable RL 方法，强化 movement / phase pressure 表示。 | 选择下一个相位 | lane pressure state + MLP Q-network | pressure | runnable approximation |
| `attendlight` | 使用 attention 处理可变数量 roads-lanes 和 phases，目标是更通用的单路口模型。 | 选择下一个相位 | lane tokens + Transformer encoder | waiting time | runnable approximation |

## Implementation Notes

当前实现先遵循“能在 TSHub + 本项目单路口接口上跑起来”的原则：

- `presslight/`: DQN + MLP，pressure reward。
- `intellilight/`: DQN + MLP，queue length reward。
- `frap/`: DQN + MLP 占位；phase competition network 后续单独补。
- `mplight/`: DQN + MLP，pressure reward；movement / phase pressure 表示后续细化。
- `attendlight/`: PPO + Transformer，以 lane token attention 近似原方法的 attention 思路。

这些实现保留论文方法的大方向，但不是逐行复现原作者代码。后续实验前应逐个
检查 state、reward、action timing 和 trainer 超参数，并在对应方法 README 中记录
和原论文的差异。

参考文献入口：

- PressLight: Learning max pressure control to coordinate traffic signals in arterial network.
- IntelliLight: A reinforcement learning approach for intelligent traffic light control.
- FRAP: Learning phase competition for traffic signal control.
- MPLight: efficient pressure-based RL traffic signal control.
- AttendLight: Universal Attention-Based Reinforcement Learning Model for Traffic Signal Control.

## Training

Use the `tshub` conda environment:

```bash
conda run -n tshub python -m tsc_algos.rl.presslight.train \
  --junction Beijing_Beihuan \
  --env_name easy_low_density \
  --total_timesteps 50000 \
  --seed 1 \
  --reward_scale 0.01 \
  --num_envs 2 \
  --vec_env subproc \
  --eval_freq 5000
```

Each method writes its latest model to:

```text
tsc_algos/rl/{method}/models/last_rl_model.zip
```

`--eval_freq` and `--checkpoint_freq` are interpreted as total timesteps. The
training scripts automatically divide them by `num_envs` for SB3 callbacks.

Training summaries are appended to:

```text
tsc_algos/rl/training_summary.csv
```

The summary records method, junction, scenario, seed, trained timesteps, elapsed
time, recent training episode statistics, and deterministic evaluation statistics
when available.

For quick debugging inside restricted environments, use `--num_envs 1 --vec_env
dummy`. For actual training, prefer `--num_envs 2` or more with `--vec_env
subproc` so each SUMO simulation runs in its own process.

DQN-based methods use `--reward_scale 0.01` by default to keep Q targets in a
stable numeric range. Raw training rewards from older runs without scaling should
not be compared directly with scaled runs.
