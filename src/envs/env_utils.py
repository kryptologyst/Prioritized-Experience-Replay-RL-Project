"""
Environment utilities and wrappers for the RL project.

This module provides utilities for working with different environments
and includes custom environment wrappers.
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
from gymnasium import spaces
import random


class ActionRepeatWrapper(gym.Wrapper):
    """
    Wrapper that repeats actions for a specified number of steps.
    
    Args:
        env: Environment to wrap
        repeat: Number of times to repeat each action
    """
    
    def __init__(self, env: gym.Env, repeat: int = 4):
        super().__init__(env)
        self.repeat = repeat
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step with action repetition."""
        total_reward = 0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class FrameStackWrapper(gym.Wrapper):
    """
    Wrapper that stacks the last k frames.
    
    Args:
        env: Environment to wrap
        k: Number of frames to stack
    """
    
    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames = []
        
        # Update observation space
        low = np.repeat(self.observation_space.low, k, axis=0)
        high = np.repeat(self.observation_space.high, k, axis=0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=self.observation_space.dtype)
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset environment and frames."""
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.k
        return np.concatenate(self.frames), info
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step and update frames."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        self.frames = self.frames[-self.k:]
        return np.concatenate(self.frames), reward, terminated, truncated, info


class RewardShapingWrapper(gym.Wrapper):
    """
    Wrapper that applies reward shaping.
    
    Args:
        env: Environment to wrap
        reward_fn: Function to transform rewards
    """
    
    def __init__(self, env: gym.Env, reward_fn: callable):
        super().__init__(env)
        self.reward_fn = reward_fn
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step with reward shaping."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = self.reward_fn(obs, reward, terminated, truncated, info)
        return obs, shaped_reward, terminated, truncated, info


class NoisyActionWrapper(gym.Wrapper):
    """
    Wrapper that adds noise to actions for continuous environments.
    
    Args:
        env: Environment to wrap
        noise_std: Standard deviation of noise
    """
    
    def __init__(self, env: gym.Env, noise_std: float = 0.1):
        super().__init__(env)
        self.noise_std = noise_std
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step with noisy action."""
        if isinstance(self.action_space, spaces.Box):
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            noisy_action = np.clip(action + noise, self.action_space.low, self.action_space.high)
        else:
            noisy_action = action
        return self.env.step(noisy_action)


def make_env(
    env_name: str,
    render_mode: Optional[str] = None,
    **kwargs
) -> gym.Env:
    """
    Create and configure an environment.
    
    Args:
        env_name: Name of the environment
        render_mode: Render mode for the environment
        **kwargs: Additional arguments for environment creation
        
    Returns:
        Configured environment
    """
    env = gym.make(env_name, render_mode=render_mode, **kwargs)
    return env


def get_env_info(env: gym.Env) -> Dict[str, Any]:
    """
    Get information about an environment.
    
    Args:
        env: Environment to analyze
        
    Returns:
        Dictionary containing environment information
    """
    info = {
        'name': env.spec.id if env.spec else 'Unknown',
        'state_dim': env.observation_space.shape[0] if len(env.observation_space.shape) == 1 else env.observation_space.shape,
        'action_dim': env.action_space.n if isinstance(env.action_space, spaces.Discrete) else env.action_space.shape[0],
        'continuous': isinstance(env.action_space, spaces.Box),
        'action_space': env.action_space,
        'observation_space': env.observation_space
    }
    
    return info


class GridWorldEnv(gym.Env):
    """
    Simple Grid World environment for testing.
    
    Args:
        size: Size of the grid (size x size)
        max_steps: Maximum steps per episode
    """
    
    def __init__(self, size: int = 8, max_steps: int = 100):
        super().__init__()
        
        self.size = size
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)
        
        # Observation space: (x, y) position
        self.observation_space = spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32)
        
        # Initialize state
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([size-1, size-1])
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        self.agent_pos = np.array([0, 0])
        self.current_step = 0
        return self.agent_pos.copy(), {}
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        self.current_step += 1
        
        # Move agent
        if action == 0:  # up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # right
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # down
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 3:  # left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
            
        # Calculate reward
        distance = np.linalg.norm(self.agent_pos - self.goal_pos)
        reward = -0.01  # Small negative reward for each step
        
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 1.0
            terminated = True
        else:
            terminated = False
            
        truncated = self.current_step >= self.max_steps
        
        return self.agent_pos.copy(), reward, terminated, truncated, {}
        
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == 'human':
            grid = np.zeros((self.size, self.size), dtype=str)
            grid.fill('.')
            grid[self.agent_pos[1], self.agent_pos[0]] = 'A'
            grid[self.goal_pos[1], self.goal_pos[0]] = 'G'
            print('\n'.join([' '.join(row) for row in grid]))
            print()
        return None


class MountainCarContinuousWrapper(gym.Wrapper):
    """
    Wrapper for MountainCarContinuous that makes it more suitable for RL.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step with shaped rewards."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Reward shaping: encourage reaching the goal faster
        position, velocity = obs
        
        # Shaped reward based on position and velocity
        shaped_reward = 0
        
        # Encourage reaching higher positions
        if position > 0.1:
            shaped_reward += 10 * position
            
        # Encourage reaching the goal
        if position >= 0.45:
            shaped_reward += 100
            
        # Small penalty for each step
        shaped_reward -= 0.1
        
        return obs, shaped_reward, terminated, truncated, info
