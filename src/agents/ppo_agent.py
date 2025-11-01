"""
Proximal Policy Optimization (PPO) Agent

This module implements a PPO agent for continuous and discrete action spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, Dict, Any
import random
from collections import deque


class PPONetwork(nn.Module):
    """
    PPO Network with shared feature extraction and separate heads for policy and value.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        continuous: Whether action space is continuous
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dims: list = [64, 64],
        continuous: bool = False
    ):
        super().__init__()
        
        self.continuous = continuous
        
        # Shared feature extraction
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh()
            ])
            input_dim = hidden_dim
            
        self.shared_layers = nn.Sequential(*layers)
        
        # Policy head
        if continuous:
            self.policy_mean = nn.Linear(input_dim, action_dim)
            self.policy_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.policy_head = nn.Linear(input_dim, action_dim)
            
        # Value head
        self.value_head = nn.Linear(input_dim, 1)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Tuple of (action_distribution, value)
        """
        features = self.shared_layers(state)
        
        # Value
        value = self.value_head(features)
        
        # Policy
        if self.continuous:
            mean = self.policy_mean(features)
            std = torch.exp(self.policy_log_std)
            action_dist = torch.distributions.Normal(mean, std)
        else:
            logits = self.policy_head(features)
            action_dist = torch.distributions.Categorical(logits=logits)
            
        return action_dist, value


class PPOAgent:
    """
    Proximal Policy Optimization Agent.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_ratio: PPO clip ratio
        value_loss_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        max_grad_norm: Maximum gradient norm for clipping
        device: Device to run on
        continuous: Whether action space is continuous
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "auto",
        continuous: bool = False,
        **kwargs
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.continuous = continuous
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Network
        self.network = PPONetwork(state_dim, action_dim, continuous=continuous).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Training state
        self.step_count = 0
        
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_dist, value = self.network(state_tensor)
            
            if training:
                action = action_dist.sample()
            else:
                if self.continuous:
                    action = action_dist.mean
                else:
                    action = action_dist.probs.argmax()
                    
            log_prob = action_dist.log_prob(action)
            
            if not self.continuous:
                action = action.item()
                log_prob = log_prob.item()
            else:
                log_prob = log_prob.sum().item()
                
        return action, log_prob, value.item()
        
    def compute_gae(
        self, 
        rewards: np.ndarray, 
        values: np.ndarray, 
        dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of done flags
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            
        returns = advantages + values
        return advantages, returns
        
    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        epochs: int = 4
    ) -> Dict[str, float]:
        """
        Update the agent using PPO.
        
        Args:
            states: Array of states
            actions: Array of actions
            old_log_probs: Array of old log probabilities
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of done flags
            epochs: Number of update epochs
            
        Returns:
            Dictionary of training metrics
        """
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device) if self.continuous else torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for _ in range(epochs):
            # Forward pass
            action_dist, values_pred = self.network(states)
            
            # Policy loss
            new_log_probs = action_dist.log_prob(actions)
            if self.continuous:
                new_log_probs = new_log_probs.sum(dim=1)
                
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values_pred.squeeze(), returns)
            
            # Entropy loss
            entropy_loss = -action_dist.entropy().mean()
            
            # Total loss
            total_loss = (
                policy_loss + 
                self.value_loss_coef * value_loss + 
                self.entropy_coef * entropy_loss
            )
            
            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            
        self.step_count += 1
        
        return {
            'policy_loss': total_policy_loss / epochs,
            'value_loss': total_value_loss / epochs,
            'entropy_loss': total_entropy_loss / epochs,
            'total_loss': (total_policy_loss + total_value_loss + total_entropy_loss) / epochs
        }
        
    def save(self, filepath: str) -> None:
        """Save the agent's state."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count
        }, filepath)
        
    def load(self, filepath: str) -> None:
        """Load the agent's state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
