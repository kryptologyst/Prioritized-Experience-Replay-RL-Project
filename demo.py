#!/usr/bin/env python3
"""
Quick demo script for the Prioritized Experience Replay RL project.

This script demonstrates the key features of the project.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demo_prioritized_replay():
    """Demonstrate prioritized experience replay."""
    print("🎯 Prioritized Experience Replay Demo")
    print("=" * 50)
    
    from src.utils.replay_buffer import PrioritizedReplayBuffer
    from src.agents.dqn_agent import DQNAgent
    from src.envs.env_utils import make_env, get_env_info
    
    # Create environment
    env = make_env("CartPole-v1")
    env_info = get_env_info(env)
    
    print(f"Environment: {env_info['name']}")
    print(f"State dimension: {env_info['state_dim']}")
    print(f"Action dimension: {env_info['action_dim']}")
    
    # Create agent with PER
    agent = DQNAgent(
        state_dim=env_info['state_dim'],
        action_dim=env_info['action_dim'],
        learning_rate=0.001,
        batch_size=32,
        buffer_size=1000,
        per_alpha=0.6,  # Prioritization strength
        per_beta=0.4,   # Importance sampling correction
        epsilon_start=0.9,
        epsilon_end=0.05,
        epsilon_decay=0.995
    )
    
    print(f"\nAgent created with device: {agent.device}")
    print(f"Replay buffer capacity: {agent.replay_buffer.capacity}")
    print(f"PER Alpha: {agent.replay_buffer.alpha}")
    print(f"PER Beta: {agent.replay_buffer.beta}")
    
    # Training loop
    print("\n🚀 Training for 100 episodes...")
    episode_rewards = []
    
    for episode in range(100):
        obs, _ = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_experience(obs, action, reward, next_obs, done)
            
            # Update agent
            loss = agent.update()
            
            obs = next_obs
            episode_reward += reward
            
            if done:
                break
                
        episode_rewards.append(episode_reward)
        
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode:3d}: Avg Reward = {avg_reward:6.2f}, Epsilon = {agent.epsilon:.3f}")
    
    # Results
    print(f"\n📊 Training Results:")
    print(f"Final average reward: {np.mean(episode_rewards[-20:]):.2f}")
    print(f"Best episode reward: {np.max(episode_rewards):.2f}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.6)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    window = 10
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    plt.plot(moving_avg, color='red', linewidth=2)
    plt.title(f'Moving Average (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
    print(f"\n📈 Results saved to 'demo_results.png'")
    
    return episode_rewards


def demo_replay_buffer():
    """Demonstrate replay buffer functionality."""
    print("\n🔄 Replay Buffer Demo")
    print("=" * 30)
    
    from src.utils.replay_buffer import PrioritizedReplayBuffer
    
    buffer = PrioritizedReplayBuffer(capacity=100)
    
    print("Adding experiences with different TD-errors...")
    
    # Add experiences with different priorities
    experiences = [
        (np.array([1, 2, 3, 4]), 0, 1.0, np.array([2, 3, 4, 5]), False, 0.1),  # Low priority
        (np.array([2, 3, 4, 5]), 1, 2.0, np.array([3, 4, 5, 6]), False, 0.8),  # High priority
        (np.array([3, 4, 5, 6]), 0, 0.5, np.array([4, 5, 6, 7]), True, 0.3),   # Medium priority
    ]
    
    for state, action, reward, next_state, done, td_error in experiences:
        buffer.add(state, action, reward, next_state, done, td_error)
        print(f"Added experience with TD-error: {td_error:.1f}")
    
    print(f"\nBuffer size: {buffer.size}")
    print(f"Priorities: {[f'{p:.3f}' for p in buffer.priorities]}")
    
    # Sample experiences
    print("\nSampling experiences...")
    sampled_experiences, indices, weights = buffer.sample(2)
    
    print(f"Sampled {len(sampled_experiences)} experiences")
    print(f"Indices: {indices}")
    print(f"Weights: {[f'{w:.3f}' for w in weights]}")
    
    # Update priorities
    print("\nUpdating priorities...")
    new_td_errors = np.array([0.5, 0.2])
    buffer.update_priorities(indices, new_td_errors)
    
    print(f"Updated priorities: {[f'{p:.3f}' for p in buffer.priorities]}")


def demo_environment():
    """Demonstrate environment functionality."""
    print("\n🌍 Environment Demo")
    print("=" * 25)
    
    from src.envs.env_utils import make_env, get_env_info, GridWorldEnv
    
    # Test standard environment
    print("Testing CartPole-v1:")
    env = make_env("CartPole-v1")
    env_info = get_env_info(env)
    
    print(f"  State dimension: {env_info['state_dim']}")
    print(f"  Action dimension: {env_info['action_dim']}")
    print(f"  Continuous actions: {env_info['continuous']}")
    
    # Test custom environment
    print("\nTesting GridWorld:")
    grid_env = GridWorldEnv(size=5)
    obs, _ = grid_env.reset()
    print(f"  Initial position: {obs}")
    
    # Take some steps
    for action in [1, 2, 1, 2]:  # right, down, right, down
        obs, reward, terminated, truncated, _ = grid_env.step(action)
        print(f"  Action {action}: Position = {obs}, Reward = {reward:.2f}")
        if terminated or truncated:
            break


def main():
    """Run the demo."""
    print("🤖 Prioritized Experience Replay RL Project Demo")
    print("=" * 60)
    
    try:
        # Demo replay buffer
        demo_replay_buffer()
        
        # Demo environment
        demo_environment()
        
        # Demo training
        episode_rewards = demo_prioritized_replay()
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ Prioritized Experience Replay Buffer")
        print("✓ DQN Agent with PER")
        print("✓ Multiple Environment Support")
        print("✓ Training Loop with Visualization")
        print("✓ Modern PyTorch Implementation")
        
        print(f"\n📁 Project Structure:")
        print("├── src/agents/     - RL agent implementations")
        print("├── src/envs/       - Environment utilities")
        print("├── src/utils/      - Replay buffer and utilities")
        print("├── config/         - Configuration files")
        print("├── notebooks/      - Jupyter notebooks")
        print("├── tests/          - Unit tests")
        print("├── train.py        - Main training script")
        print("└── streamlit_app.py - Web interface")
        
        print(f"\n🚀 Next Steps:")
        print("1. Run 'python train.py' for full training")
        print("2. Run 'streamlit run streamlit_app.py' for web UI")
        print("3. Explore notebooks/ for detailed analysis")
        print("4. Check tests/ for comprehensive testing")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
