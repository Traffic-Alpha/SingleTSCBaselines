# UniTSA

UniTSA 是在 [AttendLight](../attendlight) 基础上演进的单路口 TSC 算法。相比 AttendLight，
做了三处替换（**均已完成**）：

> Mixed-scenario v2：reward 使用决策间隔最后一帧的车辆加权平均累计等待；state 使用
> 全车道车辆数、E2 queue、累计等待和归一化绿灯计时。Transformer 输出经固定尺度
> LayerNorm 后进入 GELU policy MLP，避免旧模型中 Tanh 饱和导致的常数动作。该
> mixed-scenario 流程的训练期评估固定覆盖每个 scenario，正式模型使用训练结束保存的
> final model，不再选择随机 best；单场景 `eval.py` 仍加载 `best_model.zip`。

| 维度 | AttendLight | UniTSA（最终） |
|------|-------------|----------------|
| RL 算法 | DQN | **PPO**（独立特征提取器 + `ent_coef=0.005` + 可选 VecNormalize） |
| 动作空间 | `choose_next_phase`（选某个相位） | **`next_or_not`**（保持/切换，`Discrete(2)`） |
| Reward | movement pressure | **所有车辆的平均「累计」等待时间**（取负，抗 starvation） |

## 最终结果（Beijing_Beihuan / normal_fluctuating_commuter）

用 SUMO `tripinfo` 的真实交通指标对比（reward 已不可直接比较，必须看真实指标）。
**UniTSA 在四项指标上全部最优：**

| 算法 | 平均行程时间 | 最大行程 | **平均等待** | **最大等待** |
|------|------------:|--------:|------------:|------------:|
| **UniTSA** | **37.7** | **84** | **5.18** | **50** |
| AttendLight (DQN) | 38.6 | 114 | 5.94 | 82 |
| PressLight | 40.9 | 149 | 7.78 | 85 |
| MaxPressure | 46.9 | 134 | 11.66 | 96 |

> 注：reward 若用「平均」等待时间会被 starvation 钻空子（放行大流量、饿死小流量方向，最大
> 等待一度高达 179）；改用「累计」等待时间后，被饿死的车累计等待无界增长直接受罚，max
> waiting 降到 50。详见 Roadmap 第 5 步。

## 渐进式开发计划（Roadmap）

为了便于定位问题，按以下顺序逐步推进。**每一步只改一个变量，先确认能稳定收敛
再进入下一步**：

1. **替换算法为 PPO（已完成）** —— 复用 `choose_next_phase` 动作与 pressure reward，
   把 DQN 换成 PPO，验证收敛。已定位关键配置：`share_features_extractor=False` /
   `ent_coef=0.005` / `n_steps=256`（详见「PPO 调参记录」）。
   结果：PPO 稳定收敛到 eval -8.35，与 DQN(-8.06) 同水平。
2. **加入 VecNormalize（稳定 PPO）** —— 训练 env 外包一层 `VecNormalize`，
   **只归一化 reward（`norm_obs=False, norm_reward=True`）**，稳住 value/advantage
   尺度、降低 eval 波动。设计要点：
   - `norm_obs` 必须为 `False`：obs 已在 `state_funcs` 归一化到 [0,1]，且 Transformer
     用 `obs.abs().sum(-1)==0` 识别 padding，归一化 obs 会破坏该 mask；
   - **独立的 `eval.py` 脚本无需改动**：它不走 EvalCallback，策略只依赖未归一化的 obs，
     评估不需要 VecNormalize；
   - 但 **`train.py` 里的 `eval_env` 必须也包成 `VecNormalize`**（`norm_reward=False,
     training=False`）：训练 env 是 VecNormalize 时 EvalCallback 会
     `sync_envs_normalization(训练env, eval_env)`，要求两者同为 VecEnvWrapper，否则第一次
     eval 直接 `AssertionError` 崩溃；这样设置后 eval reward 仍是原始值；
   - 训练结束保存 `vec_normalize.pkl`（仅用于复现/续训）；`gamma` 与 PPO 一致 (0.99)；
   - 用 `--use_vecnorm` 开启（**默认关闭**）。

   **实验结论（在 pressure reward 上）**：VecNorm 实现正确、端到端可跑，但**没有带来
   改善，反而略增噪**：

   | 配置 | rollout 末5 | eval 末20 均值±std |
   |------|------------|-------------------|
   | DQN 基线 | -8.62 | -8.06 |
   | Step1 PPO（无 VecNorm） | **-8.32** | **~-8.35**（很紧） |
   | Step2 PPO（+VecNorm） | -9.52 | -10.42 ± 3.45（偶发尖刺） |

   原因：第 1 步已稳定收敛，且 reward 已被 `reward_scale=0.1` 缩放到合适范围，VecNorm
   的回报归一化再叠一层自适应缩放（running return std 漂移）反而引入非平稳噪声。
   **因此默认关闭**。但它在 **第 5 步（累计等待 reward，量纲是秒、尺度大且会随 starvation
   增长）确实是必要的并已启用** —— 印证了「VecNorm 是否有用取决于 reward 尺度」。
3. **网络改造：1D CNN 替代 Transformer（已尝试并放弃）** —— 试过"对每个 movement
   用一维 CNN（时间维卷积、F 作通道）→ 聚合"的方案，并进一步加了一层**跨 movement
   自注意力**补回 movement 间交互。结论见下表：

   | 网络 | rollout 末5 | eval 末20 均值±std | 参数量 |
   |------|------------|-------------------|--------|
   | **Transformer（保留为默认）** | **-8.32** | **-8.35**（很紧） | 319k×2 |
   | 纯 1D CNN | -12.93 | -12.22 ± 2.19 | 43.8k×2 |
   | 1D CNN + 跨 movement 注意力 | -12.98 | -11.81 ± 2.85 | 176k×2 |

   两版 CNN 都**稳定收敛但比 Transformer 差 ~3.5**（-12 vs -8.35），加跨 movement 注意力
   几乎没改善。原因：CNN 把时间维过早 mean-pool 掉，丢失时空联合结构；Transformer 对
   `(time × movement)=60` 个 token 做多层联合注意力，结构更强。再继续改 CNN 就是重造
   Transformer。且本任务 **wall-clock 瓶颈是 SUMO 仿真而非网络**，CNN 的"轻"换不到速度。
   **因此保留 Transformer 为默认，CNN 代码已删除（保持简洁）。**
4. **替换动作为 `next_or_not`（已完成，效果更好）** —— `tls_action_type` 改为
   `next_or_not`，动作空间 `Discrete(2)`（1=保持当前相位，0=切到下一相位），wrapper 改为
   `NextOrNotWrapper`；state/reward 不变。结论：

   | 动作设计 | rollout 末5 | eval 末20 均值±std |
   |---------|------------|-------------------|
   | DQN choose_next_phase | -8.62 | -8.06 |
   | PPO choose_next_phase | -8.32 | -8.35 |
   | **PPO next_or_not** | **-7.89** | **-7.59 ± 1.02**（最稳） |

   `next_or_not` 在**收敛值、稳定性、速度**三方面都更好：动作空间更小(2 vs 3)→ 探索和
   信用分配更容易；keep/switch 参数化更贴合信号控制（不用挑哪个相位，只决定是否前进）。
   熵 max 也从 ln3=1.10 变为 ln2=0.69。
5. **替换 reward 为等待时间（已完成）** —— reward 改为路口所有车辆的平均等待时间（取负），
   加到 `reward_funcs.py`。这一步踩了两个坑、各有对应修复：

   - **坑 1：用「平均瞬时」等待会被 starvation 钻空子。** reward 看似收敛（很稳），但
     tripinfo 揭穿是「假收敛」：平均等待 **35.08**、最大等待 **179**，比所有 baseline 差
     3~6 倍。因为放行大流量、压低平均值即可，而少数小流量方向的车被饿死（车少拉不动均值）。
     → **改用「累计」等待时间**（`avg_accumulated_waiting_time`，按车辆数加权）：被饿死的车
     累计等待无界增长，直接惩罚 starvation。
   - **坑 2：等待时间数值大、波动大，且确定性策略会熵塌缩到 0、停止学习。**
     → **启用 VecNormalize**（`--use_vecnorm`，只归一化 reward）稳住 value/advantage 尺度。

     | reward | 平均等待 | 最大等待 | 训练健康度 |
     |--------|--------:|--------:|-----------|
     | 平均瞬时等待 | 35.08 | 179 | 熵→0、`approx_kl→0`（塌缩） |
     | **累计等待 + VecNorm** | **5.18** | **50** | 熵 0.12、`approx_kl=0.042`（仍在学） |

   修复后 UniTSA 在真实指标上**超过所有 baseline**（见上文「最终结果」）。注意：VecNorm 在
   第 2 步的 pressure reward 上无增益，但在这里（大尺度累计等待 reward）是必要的 —— 印证了
   它的用武之地取决于 reward 尺度。
6. **最终 UniTSA（已完成）** —— 整合 **PPO（独立主干 + `ent_coef=0.005` + `n_steps=256`）
   + `next_or_not` 动作 + 累计等待 reward + VecNormalize**，端到端训练稳定收敛，并在 SUMO
   tripinfo 真实指标（平均/最大 等待与行程时间）上**全面优于 AttendLight(DQN) / PressLight
   / MaxPressure**。

## 目录结构

```
unitsa/
  train.py                 # PPO 训练入口（VecNormalize 内置常开）
  eval.py                  # PPO 评估入口（--gui 可选；默认 headless 输出 tripinfo）
  model.py                 # UniTSAMovementTransformer（movement-token Transformer）
  unitsa_env/              # 环境组装
    make_env.py            # TSCEnvironment(next_or_not) -> TSCInfoWrapper -> NextOrNotWrapper -> Monitor
    rl_wrapper.py          # NextOrNotWrapper：动作空间 Discrete(2)
    state_funcs.py         # movement-level 时序 state
    reward_funcs.py        # waiting_time_reward（累计等待，取负）
  checkpoints/             # 训练产物（best_model.zip / vec_normalize.pkl），已 gitignore
```

## 运行

训练（沙箱内用 `dummy`，本机可用 `subproc` 并行）。下面是 **UniTSA 最终配置**
（累计等待 reward + VecNormalize；调参依据见后文「PPO 调参记录」）：

```bash
conda run -n tshub python tsc_algos/rl/unitsa/train.py \
    --junction Beijing_Beihuan --env_name normal_fluctuating_commuter \
    --num_envs 20 --vec_env subproc --reward_scale 0.1 --history_len 5 \
    --n_steps 256 --batch_size 256 --n_epochs 10 --ent_coef 0.005
```

（reward 固定为累计等待时间、VecNormalize 内置常开，无需额外开关。）

评估（默认 headless，输出 SUMO `tripinfo` 到 `tsc_algos/results/unitsa/...`，
用 `--gui` 可开界面）：

```bash
conda run -n tshub python tsc_algos/rl/unitsa/eval.py \
    --junction Beijing_Beihuan --env_name normal_fluctuating_commuter --history_len 5
```

对比真实指标：用 `tshub` 的 `TripInfoAnalysis`（参考 `tsc_algos/results/analysis_tripinfo.py`）
读取各算法的 `trip_info.xml`，比较 `duration` / `waitingTime`。

## PPO 调参记录（DQN → PPO 第 1 步的踩坑与结论）

把 AttendLight 的 DQN 直接换成 PPO 后，PPO **完全学不动**（reward 几乎不降）。
同一套 env / reward / 网络下 DQN 能收敛（~22 episode/env 时 reward 从 -40 →
-12），所以问题在 **PPO 侧的配置**，不是环境。逐变量定位后结论如下。

### 决定性根因：actor / critic 不能共享特征提取器

PPO policy 的参数分布（obs `(T=5, M=12, F=12)`，action `Discrete(3)`）：

| 模块 | 参数量 | 说明 |
|------|------:|------|
| **Transformer 主干**（features extractor） | **319,232** | 占 ~93%，策略的「大脑」 |
| mlp_extractor（net_arch 64×64） | 24,832 | |
| **action_net（策略头）** | **195** | 仅 64→3，极小 |
| value_net | 65 | 64→1 |

- SB3 默认 `share_features_extractor=True`：**一个**主干被 actor 和 critic 共用，
  其梯度 = ∇policy_loss + `vf_coef`·∇value_loss。`value_loss` 量级大（~7–9）且
  回归信号「好学」，于是**主干被优化成预测状态价值（与动作无关）**，仅 195 参数
  的 action_net 抽不出动作偏好 → **策略熵卡在最大值 ln(3)=1.099，始终均匀随机**。
- 这也解释了**为什么 DQN 行、PPO 共享时不行**：DQN 的 Q-loss 天生按动作分头、且
  有 replay 反复更新，主干能直接拿到强动作梯度；PPO 的策略梯度只能透过 195 参数
  小头回传，被 value 梯度淹没。

**解决：`policy_kwargs` 设 `share_features_extractor=False`**，让 actor 拥有
独立主干（代价：两个 Transformer，参数 344k → 664k）。改后熵稳步下降、
`eval/mean_reward` 从 -335 一路降到 ~-28，**开始收敛**。

### `ent_coef`：不是根因，但需要一个小的非零值

先尝试把 `ent_coef` 从 0.01 调到 0（去掉熵奖励对均匀分布的拉力）。但在**仍共享
主干**时，单独改 ent_coef **无效**，熵照样卡在 ln(3)。说明**根因是共享主干而非熵
正则**（当初 0.01 学不动 = 共享主干让策略拿不到梯度，被 value 梯度淹没）。

独立主干 + `ent_coef=0` 能学，但**确定性 eval 间歇塌缩**：argmax 偶尔退化成
「全程切相位」的恒定动作策略 → episode 拖到长度上限(200)、reward 掉到 -330，
拉垮 `eval/mean_reward` 均值（详见下文 episode length）。原因是 `ent_coef=0` 没有
熵下限，确定性策略容易过度 commit / 塌缩。

最终定为 **`ent_coef=0.005`**（务必配 `share_features_extractor=False`）：
- 给一个**熵下限**防止确定性策略塌缩；
- 取 0.005 而非 0.01，是因为本任务 advantage 信号偏弱（ent_coef=0 时熵仅温和下降
  1.09→0.90），0.01 的熵奖励可能把这点 commit 压回去；0.005 防塌缩又不压学习。

### `n_steps`：太小会再次破坏收敛

`rollout = n_steps × num_envs`；总更新次数 = `total_timesteps / rollout`。两者对立：

| n_steps | rollout | 每环境段长 vs 一个 ep(~143) | 总更新次数 | 优势质量 | 结果 |
|--------:|-------:|------|--------:|------|------|
| 64  | 1280  | 0.45×（碎片） | 234 | 噪声大 | ❌ 熵又卡死 |
| **128** | 2560 | 0.9×（≈一个 ep） | 117 | 尚可 | 提速候选 |
| **256** | 5120 | 1.8× | 58 | 干净 | ✅ 已验证收敛 |
| 512 | 10240 | 3.6× | 29 | 很干净 | 更稳但更新少→更慢 |

要点：训练这个 **319k 大策略主干**需要**低方差的 GAE 优势**，门槛约为「每环境
rollout 段长 ≥ 一个 episode(~145)」。`n_steps=64` 段长远小于一个 episode →
优势太噪 → 大主干拿到噪声梯度 → **又卡回均匀**。所以「更小 n_steps = 更快」在此
是错的；`n_steps` 应 ≥ 256。512 收敛更稳但更新仅 29 次，通常更慢。

### 一个易被忽视的混淆量：episode length

- 训练 rollout（随机策略）`ep_len ≈ 143`；eval（确定性 argmax）`ep_len` 可达 200。
- 由于动作（切相位频率）会改变一个仿真时段内的**决策步数**，而每步 reward 为负，
  **总 episode reward 被长度混淆**。对比策略好坏时，更可靠的是看**每步 reward**
  或排队 / 延误指标，而非总回报。

### 顺带修复的共享 bug

`tsc_env/base_env.py` 之前把空字符串 `trip_info`/`fcd_output` 直接传给 SUMO，新版
libsumo 会因空输出路径报 `Could not build output file ''`，导致**训练环境一启动就
崩**（AttendLight 同样受影响）。已改为 `trip_info or None`，空串视为不输出。

### 关键超参速查（已验证收敛）

```
share_features_extractor = False   # 决定性：actor/critic 各自独立主干
ent_coef                 = 0.005   # 小熵下限：防确定性策略塌缩，又不压弱 advantage
n_steps                  = 256     # 保证 GAE 优势干净（≥ 一个 episode）
batch_size               = 256
n_epochs                 = 10
learning_rate            = 3e-4    # 提速可试 5e-4，但保持 n_steps ≥ 256
```
