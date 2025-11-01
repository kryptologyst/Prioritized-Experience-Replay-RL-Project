"""
Unit tests for the RL project.

This module contains tests for the core components of the project.
"""

import pytest
import numpy as np
import torch
import gymnasium as gym

from src.utils.replay_buffer import PrioritizedReplayBuffer, Experience
from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.sac_agent import SACAgent
from src.envs.env_utils import GridWorldEnv, get_env_info


class TestPrioritizedReplayBuffer:
    """Test cases for PrioritizedReplayBuffer."""
    
    def test_buffer_initialization(self):
        """Test buffer initialization."""
        buffer = PrioritizedReplayBuffer(capacity=1000)
        assert buffer.capacity == 1000
        assert buffer.size == 0
        assert len(buffer) == 0
        
    def test_add_experience(self):
        """Test adding experiences to buffer."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        # Add some experiences
        for i in range(10):
            state = np.random.randn(4)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = i % 5 == 0
            
            buffer.add(state, action, reward, next_state, done)
            
        assert buffer.size == 10
        assert len(buffer) == 10
        
    def test_sample_experiences(self):
        """Test sampling experiences from buffer."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        # Add experiences
        for i in range(50):
            state = np.random.randn(4)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = i % 10 == 0
            
            buffer.add(state, action, reward, next_state, done)
            
        # Sample batch
        experiences, indices, weights = buffer.sample(32)
        
        assert len(experiences) == 32
        assert len(indices) == 32
        assert len(weights) == 32
        assert all(isinstance(exp, Experience) for exp in experiences)
        
    def test_update_priorities(self):
        """Test updating priorities."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        # Add experiences
        for i in range(20):
            state = np.random.randn(4)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = i % 5 == 0
            
            buffer.add(state, action, reward, next_state, done)
            
        # Update priorities
        indices = np.array([0, 5, 10])
        td_errors = np.array([0.5, 1.0, 0.3])
        
        buffer.update_priorities(indices, td_errors)
        
        # Check that priorities were updated
        assert buffer.priorities[0] == abs(0.5) + buffer.epsilon
        assert buffer.priorities[5] == abs(1.0) + buffer.epsilon
        assert buffer.priorities[10] == abs(0.3) + buffer.epsilon


class TestDQNAgent:
    """Test cases for DQNAgent."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = DQNAgent(state_dim=4, action_dim=2)
        
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert agent.device is not None
        
    def test_action_selection(self):
        """Test action selection."""
        agent = DQNAgent(state_dim=4, action_dim=2)
        state = np.random.randn(4)
        
        action = agent.select_action(state)
        assert isinstance(action, int)
        assert 0 <= action < 2
        
    def test_experience_storage(self):
        """Test experience storage."""
        agent = DQNAgent(state_dim=4, action_dim=2)
        
        state = np.random.randn(4)
        action = 1
        reward = 1.0
        next_state = np.random.randn(4)
        done = False
        
        agent.store_experience(state, action, reward, next_state, done)
        assert agent.replay_buffer.size == 1
        
    def test_agent_update(self):
        """Test agent update."""
        agent = DQNAgent(state_dim=4, action_dim=2, batch_size=16)
        
        # Add enough experiences for batch
        for i in range(20):
            state = np.random.randn(4)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = i % 10 == 0
            
            agent.store_experience(state, action, reward, next_state, done)
            
        # Update agent
        loss = agent.update()
        assert loss is not None
        assert isinstance(loss, float)


class TestPPOAgent:
    """Test cases for PPOAgent."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = PPOAgent(state_dim=4, action_dim=2)
        
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert agent.device is not None
        
    def test_action_selection(self):
        """Test action selection."""
        agent = PPOAgent(state_dim=4, action_dim=2)
        state = np.random.randn(4)
        
        action, log_prob, value = agent.select_action(state)
        assert isinstance(action, int)
        assert 0 <= action < 2
        assert isinstance(log_prob, float)
        assert isinstance(value, float)


class TestSACAgent:
    """Test cases for SACAgent."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = SACAgent(state_dim=4, action_dim=2)
        
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert agent.device is not None
        
    def test_action_selection(self):
        """Test action selection."""
        agent = SACAgent(state_dim=4, action_dim=2)
        state = np.random.randn(4)
        
        action = agent.select_action(state)
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)


class TestGridWorldEnv:
    """Test cases for GridWorldEnv."""
    
    def test_env_initialization(self):
        """Test environment initialization."""
        env = GridWorldEnv(size=8)
        
        assert env.size == 8
        assert env.action_space.n == 4
        assert env.observation_space.shape == (2,)
        
    def test_env_reset(self):
        """Test environment reset."""
        env = GridWorldEnv(size=8)
        
        obs, info = env.reset()
        assert obs.shape == (2,)
        assert np.array_equal(obs, [0, 0])  # Agent starts at (0, 0)
        assert env.current_step == 0
        
    def test_env_step(self):
        """Test environment step."""
        env = GridWorldEnv(size=8)
        obs, _ = env.reset()
        
        # Take a step
        next_obs, reward, terminated, truncated, info = env.step(1)  # Move right
        
        assert next_obs.shape == (2,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert env.current_step == 1


class TestEnvUtils:
    """Test cases for environment utilities."""
    
    def test_get_env_info(self):
        """Test getting environment information."""
        env = gym.make("CartPole-v1")
        info = get_env_info(env)
        
        assert 'name' in info
        assert 'state_dim' in info
        assert 'action_dim' in info
        assert 'continuous' in info
        assert info['continuous'] == False  # CartPole has discrete actions
        
    def test_make_env(self):
        """Test creating environment."""
        env = gym.make("CartPole-v1")
        
        assert isinstance(env, gym.Env)
        assert env.observation_space is not None
        assert env.action_space is not None


if __name__ == "__main__":
    pytest.main([__file__])
