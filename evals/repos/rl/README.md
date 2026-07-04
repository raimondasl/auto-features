# PolicyKit

A deep reinforcement-learning library of reliable, well-tested algorithm
implementations. PolicyKit provides on-policy and off-policy agents with a
consistent API, designed for research reproducibility and easy benchmarking on
continuous-control and Atari environments.

## Features

- **Policy-gradient methods** — PPO, A2C, and TRPO with generalized advantage
  estimation.
- **Off-policy actor-critic** — SAC, TD3, and DDPG with replay buffers and target
  networks.
- **Value-based agents** — DQN with double-Q, dueling heads, and prioritized
  experience replay.
- **Vectorized environments** — parallel rollout collection over Gymnasium envs.
- **Reproducible benchmarks** — deterministic seeding and logged training curves.

## Quick start

```python
from policykit import PPO

agent = PPO("MlpPolicy", "CartPole-v1")
agent.learn(total_timesteps=100_000)
agent.save("ppo_cartpole")
```

PolicyKit targets deep reinforcement-learning research on policy optimization and
sample-efficient control.
