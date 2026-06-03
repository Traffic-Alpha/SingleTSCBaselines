# FRAP

Status: scaffold, phase competition network pending.

Current implementation:

- Trainer: Stable-Baselines3 DQN.
- Action: choose next phase.
- State: lane aggregate placeholder.
- Reward: queue length reward.
- Model: MLP placeholder.

Todo for stricter reproduction:

- Implement phase-pair state construction.
- Replace the MLP placeholder with a phase competition network.
- Encode conflict relations between phases explicitly.
