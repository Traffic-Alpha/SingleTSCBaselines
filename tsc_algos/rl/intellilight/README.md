# IntelliLight

Status: runnable approximation.

Current implementation:

- Trainer: Stable-Baselines3 DQN.
- Action: choose next phase.
- State: lane aggregate features with current green and phase binding.
- Reward: queue length reward.
- Model: MLP Q-network feature extractor.

Todo for stricter reproduction:

- Align the state with IntelliLight's interpretable traffic indicators.
- Check whether the original action is phase selection or switch/keep in each experiment.
- Add paper-aligned reward and DQN hyperparameters.
