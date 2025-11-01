#!/usr/bin/env python3
"""
Simple test script to verify the RL project works correctly.

This script tests the basic functionality of the project components.
"""

import sys
import os
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.utils.replay_buffer import PrioritizedReplayBuffer, Experience
        print("✓ Replay buffer imports successful")
    except ImportError as e:
        print(f"✗ Replay buffer import failed: {e}")
        return False
        
    try:
        from src.agents.dqn_agent import DQNAgent
        print("✓ DQN agent import successful")
    except ImportError as e:
        print(f"✗ DQN agent import failed: {e}")
        return False
        
    try:
        from src.agents.ppo_agent import PPOAgent
        print("✓ PPO agent import successful")
    except ImportError as e:
        print(f"✗ PPO agent import failed: {e}")
        return False
        
    try:
        from src.agents.sac_agent import SACAgent
        print("✓ SAC agent import successful")
    except ImportError as e:
        print(f"✗ SAC agent import failed: {e}")
        return False
        
    try:
        from src.envs.env_utils import make_env, get_env_info, GridWorldEnv
        print("✓ Environment utils import successful")
    except ImportError as e:
        print(f"✗ Environment utils import failed: {e}")
        return False
        
    return True


def test_replay_buffer():
    """Test replay buffer functionality."""
    print("\nTesting replay buffer...")
    
    try:
        from src.utils.replay_buffer import PrioritizedReplayBuffer
        
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        # Add some experiences
        for i in range(10):
            state = np.random.randn(4)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = i % 5 == 0
            
            buffer.add(state, action, reward, next_state, done)
            
        assert buffer.size == 10, f"Expected size 10, got {buffer.size}"
        
        # Sample batch
        experiences, indices, weights = buffer.sample(5)
        assert len(experiences) == 5, f"Expected 5 experiences, got {len(experiences)}"
        assert len(indices) == 5, f"Expected 5 indices, got {len(indices)}"
        assert len(weights) == 5, f"Expected 5 weights, got {len(weights)}"
        
        print("✓ Replay buffer tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Replay buffer test failed: {e}")
        return False


def test_dqn_agent():
    """Test DQN agent functionality."""
    print("\nTesting DQN agent...")
    
    try:
        from src.agents.dqn_agent import DQNAgent
        
        agent = DQNAgent(state_dim=4, action_dim=2)
        
        # Test action selection
        state = np.random.randn(4)
        action = agent.select_action(state)
        assert isinstance(action, int), f"Expected int action, got {type(action)}"
        assert 0 <= action < 2, f"Action {action} out of range [0, 2)"
        
        # Test experience storage
        agent.store_experience(state, action, 1.0, state, False)
        assert agent.replay_buffer.size == 1, f"Expected buffer size 1, got {agent.replay_buffer.size}"
        
        print("✓ DQN agent tests passed")
        return True
        
    except Exception as e:
        print(f"✗ DQN agent test failed: {e}")
        return False


def test_environment():
    """Test environment functionality."""
    print("\nTesting environment...")
    
    try:
        import gymnasium as gym
        from src.envs.env_utils import make_env, get_env_info, GridWorldEnv
        
        # Test standard environment
        env = make_env("CartPole-v1")
        env_info = get_env_info(env)
        
        assert env_info['state_dim'] == 4, f"Expected state dim 4, got {env_info['state_dim']}"
        assert env_info['action_dim'] == 2, f"Expected action dim 2, got {env_info['action_dim']}"
        assert not env_info['continuous'], "Expected discrete actions"
        
        # Test custom environment
        grid_env = GridWorldEnv(size=8)
        obs, _ = grid_env.reset()
        assert obs.shape == (2,), f"Expected obs shape (2,), got {obs.shape}"
        
        action = 1  # Move right
        next_obs, reward, terminated, truncated, _ = grid_env.step(action)
        assert next_obs.shape == (2,), f"Expected next_obs shape (2,), got {next_obs.shape}"
        
        print("✓ Environment tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False


def test_training_loop():
    """Test a simple training loop."""
    print("\nTesting training loop...")
    
    try:
        from src.agents.dqn_agent import DQNAgent
        from src.envs.env_utils import make_env
        
        env = make_env("CartPole-v1")
        agent = DQNAgent(state_dim=4, action_dim=2, batch_size=16)
        
        # Run a few episodes
        for episode in range(3):
            obs, _ = env.reset()
            done = False
            
            while not done:
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.store_experience(obs, action, reward, next_obs, done)
                
                # Update if we have enough experiences
                if agent.replay_buffer.size >= agent.batch_size:
                    loss = agent.update()
                    if loss is not None:
                        assert isinstance(loss, float), f"Expected float loss, got {type(loss)}"
                
                obs = next_obs
                
        print("✓ Training loop tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Training loop test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Running RL Project Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_replay_buffer,
        test_dqn_agent,
        test_environment,
        test_training_loop
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The project is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
