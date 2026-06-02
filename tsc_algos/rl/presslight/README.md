# PressLight

Status: runnable approximation.

Current implementation:

- Trainer: Stable-Baselines3 DQN.
- Action: choose next phase.
- State: fixed-length movement time series built from TLS features, vehicle count, current phase, direction, and phase binding.
- Reward: weighted time-series movement pressure reward over controllable movements.
- Model: shared movement-row MLP encoder, mean/max movement pooling, and GRU temporal aggregation.

Todo for stricter reproduction:

- Compare reward/action timing with the original setting.
- Add paper-aligned hyperparameters after short sanity experiments.
