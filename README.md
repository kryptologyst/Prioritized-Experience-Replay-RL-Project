# Prioritized Experience Replay RL Project

A comprehensive reinforcement learning project implementing prioritized experience replay with state-of-the-art algorithms including DQN, PPO, SAC, and TD3.

## Features

- **Multiple RL Algorithms**: DQN, Rainbow DQN, PPO, SAC, TD3
- **Prioritized Experience Replay**: Efficient learning from important experiences
- **Modern Libraries**: Gymnasium, PyTorch, Stable-Baselines3, Ray RLLib
- **Interactive UI**: Streamlit web interface for training and visualization
- **Comprehensive Logging**: TensorBoard and Weights & Biases integration
- **Multiple Environments**: CartPole, MountainCar, LunarLander, GridWorld
- **Type Hints & Documentation**: Full type annotations and docstrings
- **Configuration System**: YAML-based configuration management
- **Checkpointing**: Save and resume training from checkpoints

## 📁 Project Structure

```
├── src/
│   ├── agents/           # RL agent implementations
│   │   ├── dqn_agent.py
│   │   ├── ppo_agent.py
│   │   └── sac_agent.py
│   ├── envs/            # Environment utilities and wrappers
│   │   └── env_utils.py
│   └── utils/           # Utility modules
│       ├── replay_buffer.py
│       └── logger.py
├── config/              # Configuration files
│   └── default.yaml
├── notebooks/           # Jupyter notebooks for analysis
├── tests/              # Unit tests
├── logs/               # Training logs and checkpoints
├── train.py            # Main training script
├── streamlit_app.py    # Streamlit web interface
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Prioritized-Experience-Replay-RL-Project.git
   cd Prioritized-Experience-Replay-RL-Project
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Command Line Training

Train a DQN agent on CartPole with prioritized experience replay:

```bash
python train.py --config config/default.yaml --log_dir logs/cartpole_dqn
```

### Streamlit Web Interface

Launch the interactive web interface:

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501` to:
- Configure training parameters
- Select environments and agents
- Monitor training progress in real-time
- Visualize results

### Jupyter Notebooks

Explore the project interactively:

```bash
jupyter notebook notebooks/
```

## 🔧 Configuration

The project uses YAML configuration files. Key parameters:

```yaml
# Environment settings
env:
  name: "CartPole-v1"
  max_episode_steps: 500

# Agent settings
agent:
  type: "DQN"  # Options: DQN, RainbowDQN, PPO, SAC, TD3
  learning_rate: 0.001
  batch_size: 32

# Prioritized Experience Replay
per:
  alpha: 0.6    # Prioritization strength
  beta: 0.4     # Importance sampling correction
  epsilon: 1e-6 # Small constant for priorities

# Training settings
training:
  total_timesteps: 100000
  eval_frequency: 10000
  save_frequency: 50000
```

## Supported Agents

### Deep Q-Network (DQN)
- Standard DQN with prioritized experience replay
- Double DQN for improved stability
- Dueling architecture support
- Distributional RL (Rainbow DQN)

### Proximal Policy Optimization (PPO)
- On-policy algorithm for both discrete and continuous actions
- Generalized Advantage Estimation (GAE)
- Clipped objective function

### Soft Actor-Critic (SAC)
- Off-policy algorithm for continuous action spaces
- Automatic temperature tuning
- Twin critics for improved stability

## Supported Environments

- **CartPole-v1**: Classic control task
- **MountainCar-v0**: Sparse reward environment
- **MountainCarContinuous-v0**: Continuous action version
- **LunarLander-v2**: Complex control task
- **GridWorld**: Custom discrete environment

## Monitoring and Visualization

### TensorBoard
```bash
tensorboard --logdir logs/
```

### Weights & Biases
Enable in configuration:
```yaml
logging:
  use_wandb: true
  project: "rl-prioritized-replay"
```

### Streamlit Dashboard
Real-time training visualization with:
- Episode rewards and lengths
- Moving averages
- Reward distributions
- Environment visualization

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## Performance

The prioritized experience replay implementation provides:
- **Faster convergence** compared to uniform sampling
- **Better sample efficiency** by focusing on important experiences
- **Improved stability** with importance sampling correction

## Research Features

- **Sum Tree Implementation**: O(log n) sampling complexity
- **Importance Sampling**: Bias correction for prioritized sampling
- **Multiple Prioritization Methods**: TD-error based and rank-based
- **Distributional RL**: Support for Rainbow DQN
- **Continuous Action Spaces**: SAC and TD3 implementations

## Usage Examples

### Basic Training
```python
from src.agents.dqn_agent import DQNAgent
from src.envs.env_utils import make_env

# Create environment and agent
env = make_env("CartPole-v1")
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    per_alpha=0.6,
    per_beta=0.4
)

# Training loop
for episode in range(1000):
    obs, _ = env.reset()
    done = False
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.store_experience(obs, action, reward, next_obs, done)
        agent.update()
        obs = next_obs
```

### Custom Environment
```python
from src.envs.env_utils import GridWorldEnv

# Create custom GridWorld environment
env = GridWorldEnv(size=10, max_steps=200)
agent = DQNAgent(
    state_dim=2,  # (x, y) position
    action_dim=4,  # up, right, down, left
    per_alpha=0.6
)
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest tests/`
6. Commit your changes: `git commit -m "Add feature"`
7. Push to the branch: `git push origin feature-name`
8. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gym/Gymnasium for the environment framework
- PyTorch team for the deep learning framework
- The RL research community for the algorithms and techniques
- Streamlit team for the web interface framework

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in the `notebooks/` directory
- Review the configuration examples in `config/`


# Prioritized-Experience-Replay-RL-Project
