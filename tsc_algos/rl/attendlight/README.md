# AttendLight

Status: runnable approximation.

Current implementation:

- Trainer: Stable-Baselines3 PPO.
- Action: choose next phase.
- State: lane aggregate features as lane tokens.
- Reward: waiting time reward.
- Model: Transformer feature extractor as an attention-based approximation.

Todo for stricter reproduction:

- Split the model into road-lane attention and phase attention modules.
- Support variable lane / phase counts more explicitly.
- Check the original reward and policy optimization settings.
