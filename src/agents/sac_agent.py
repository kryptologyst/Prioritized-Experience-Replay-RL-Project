"""
Soft Actor-Critic (SAC) Agent

This module implements a SAC agent for continuous action spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, Dict, Any
import random
from collections import deque


class SACActor(nn.Module):
    """
    SAC Actor network.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        max_action: Maximum action value
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dims: list = [256, 256],
        max_action: float = 1.0
    ):
        super().__init__()
        
        self.max_action = max_action
        
        # Shared layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
            
        self.shared_layers = nn.Sequential(*layers)
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Tuple of (action, log_prob)
        """
        features = self.shared_layers(state)
        
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, -20, 2)
        
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.max_action
        
        # Log probability with tanh transformation
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - torch.tanh(x_t) ** 2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob


class SACCritic(nn.Module):
    """
    SAC Critic network (Q-function).
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dims: list = [256, 256]
    ):
        super().__init__()
        
        # Input is state + action
        input_dim = state_dim + action_dim
        
        layers = []
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class SACAgent:
    """
    Soft Actor-Critic Agent.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        tau: Soft update coefficient
        alpha: Temperature parameter (if None, will be learned)
        buffer_size: Size of replay buffer
        batch_size: Batch size for training
        device: Device to run on
        max_action: Maximum action value
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: Optional[float] = None,
        buffer_size: int = 100000,
        batch_size: int = 256,
        device: str = "auto",
        max_action: float = 1.0,
        **kwargs
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Networks
        self.actor = SACActor(state_dim, action_dim, max_action=max_action).to(self.device)
        self.critic1 = SACCritic(state_dim, action_dim).to(self.device)
        self.critic2 = SACCritic(state_dim, action_dim).to(self.device)
        
        # Target networks
        self.critic1_target = SACCritic(state_dim, action_dim).to(self.device)
        self.critic2_target = SACCritic(state_dim, action_dim).to(self.device)
        
        # Copy weights to target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        
        # Temperature parameter
        if alpha is None:
            self.learn_alpha = True
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        else:
            self.learn_alpha = False
            self.alpha = alpha
            
        # Replay buffer
        from ..utils.replay_buffer import PrioritizedReplayBuffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            alpha=kwargs.get('per_alpha', 0.6),
            beta=kwargs.get('per_beta', 0.4),
            beta_increment=kwargs.get('per_beta_increment', 0.001),
            epsilon=kwargs.get('per_epsilon', 1e-6)
        )
        
        # Training state
        self.step_count = 0
        
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if training:
                action, _ = self.actor(state_tensor)
            else:
                # Use mean action for evaluation
                features = self.actor.shared_layers(state_tensor)
                mean = self.actor.mean_head(features)
                action = torch.tanh(mean) * self.max_action
                
        return action.cpu().numpy().flatten()
        
    def store_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store experience in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
        
    def update(self) -> Optional[Dict[str, float]]:
        """
        Update the agent using SAC.
        
        Returns:
            Dictionary of training metrics if training occurred, None otherwise
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
            
        # Sample batch
        experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([exp.state for exp in experiences]).to(self.device)
        actions = torch.FloatTensor([exp.action for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current temperature
        if self.learn_alpha:
            alpha = torch.exp(self.log_alpha)
        else:
            alpha = self.alpha
            
        # Critic update
        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            target_q = rewards.unsqueeze(1) + self.gamma * q_next * (~dones).unsqueeze(1)
            
        # Current Q-values
        q1_current = self.critic1(states, actions)
        q2_current = self.critic2(states, actions)
        
        # Critic losses
        critic1_loss = F.mse_loss(q1_current, target_q, reduction='none')
        critic2_loss = F.mse_loss(q2_current, target_q, reduction='none')
        
        # Weighted losses
        critic1_loss = (critic1_loss * weights.unsqueeze(1)).mean()
        critic2_loss = (critic2_loss * weights.unsqueeze(1)).mean()
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Actor update
        new_actions, log_probs = self.actor(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Temperature update
        if self.learn_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + 1).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
        # Soft update target networks
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)
        
        # Update priorities
        td_errors = torch.abs(q1_current - target_q).squeeze().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Update beta
        self.replay_buffer.update_beta()
        
        self.step_count += 1
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': alpha.item() if self.learn_alpha else alpha
        }
        
    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        """Soft update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
    def save(self, filepath: str) -> None:
        """Save the agent's state."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.learn_alpha else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.learn_alpha else None,
            'step_count': self.step_count,
            'replay_buffer': self.replay_buffer
        }, filepath)
        
    def load(self, filepath: str) -> None:
        """Load the agent's state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        
        if self.learn_alpha and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            
        self.step_count = checkpoint['step_count']
        self.replay_buffer = checkpoint['replay_buffer']
