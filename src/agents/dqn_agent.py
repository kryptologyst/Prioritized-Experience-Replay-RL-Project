"""
Deep Q-Network (DQN) Agent with Prioritized Experience Replay

This module implements a DQN agent that uses prioritized experience replay
for more efficient learning from important experiences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, Dict, Any
import random
from collections import deque

from ..utils.replay_buffer import PrioritizedReplayBuffer, Experience


class DQNNetwork(nn.Module):
    """
    Deep Q-Network architecture.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class RainbowDQNNetwork(nn.Module):
    """
    Rainbow DQN network with distributional RL and dueling architecture.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        n_atoms: Number of atoms for distributional RL
        v_min: Minimum value for distributional RL
        v_max: Maximum value for distributional RL
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dims: list = [128, 128],
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0
    ):
        super().__init__()
        
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Shared feature extraction
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU()
        )
        
        # Dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], n_atoms)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim * n_atoms)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dueling architecture."""
        features = self.feature_layers(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Reshape advantage
        advantage = advantage.view(-1, self.n_atoms, advantage.size(-1) // self.n_atoms)
        
        # Combine value and advantage
        q_dist = value.unsqueeze(-1) + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return F.softmax(q_dist, dim=1)


class DQNAgent:
    """
    DQN Agent with Prioritized Experience Replay.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        epsilon_start: Starting epsilon for exploration
        epsilon_end: Final epsilon for exploration
        epsilon_decay: Epsilon decay rate
        buffer_size: Size of replay buffer
        batch_size: Batch size for training
        target_update_frequency: How often to update target network
        device: Device to run on
        use_double_dqn: Whether to use Double DQN
        use_dueling: Whether to use dueling architecture
        use_distributional: Whether to use distributional RL
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_frequency: int = 1000,
        device: str = "auto",
        use_double_dqn: bool = True,
        use_dueling: bool = False,
        use_distributional: bool = False,
        **kwargs
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling
        self.use_distributional = use_distributional
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Networks - filter out PER parameters
        network_kwargs = {k: v for k, v in kwargs.items() if not k.startswith('per_')}
        
        if use_distributional:
            self.q_network = RainbowDQNNetwork(state_dim, action_dim, **network_kwargs).to(self.device)
            self.target_network = RainbowDQNNetwork(state_dim, action_dim, **network_kwargs).to(self.device)
        elif use_dueling:
            self.q_network = DQNNetwork(state_dim, action_dim, **network_kwargs).to(self.device)
            self.target_network = DQNNetwork(state_dim, action_dim, **network_kwargs).to(self.device)
        else:
            self.q_network = DQNNetwork(state_dim, action_dim, **network_kwargs).to(self.device)
            self.target_network = DQNNetwork(state_dim, action_dim, **network_kwargs).to(self.device)
            
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            alpha=kwargs.get('per_alpha', 0.6),
            beta=kwargs.get('per_beta', 0.4),
            beta_increment=kwargs.get('per_beta_increment', 0.001),
            epsilon=kwargs.get('per_epsilon', 1e-6)
        )
        
        # Training state
        self.epsilon = epsilon_start
        self.step_count = 0
        self.episode_rewards = deque(maxlen=100)
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            if self.use_distributional:
                # For distributional RL, use expected value
                q_values = torch.sum(q_values * torch.linspace(
                    self.q_network.v_min, 
                    self.q_network.v_max, 
                    self.q_network.n_atoms
                ).to(self.device), dim=1)
                
        return q_values.argmax().item()
        
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: Optional[float] = None
    ) -> None:
        """Store experience in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done, td_error)
        
    def update(self) -> Optional[float]:
        """
        Update the agent using prioritized experience replay.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
            
        # Sample batch
        experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array([exp.state for exp in experiences])).to(self.device)
        actions = torch.LongTensor(np.array([exp.action for exp in experiences])).to(self.device)
        rewards = torch.FloatTensor(np.array([exp.reward for exp in experiences])).to(self.device)
        next_states = torch.FloatTensor(np.array([exp.next_state for exp in experiences])).to(self.device)
        dones = torch.BoolTensor(np.array([exp.done for exp in experiences])).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use main network to select action, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
                
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (~dones).unsqueeze(1)
            
        # Calculate TD-errors
        td_errors = (target_q_values - current_q_values).squeeze().detach().cpu().numpy()
        
        # Calculate loss with importance sampling weights
        loss = F.mse_loss(current_q_values, target_q_values, reduction='none')
        weighted_loss = (loss * weights.unsqueeze(1)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()
        
        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        # Update exploration
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update beta for importance sampling
        self.replay_buffer.update_beta()
        
        return weighted_loss.item()
        
    def save(self, filepath: str) -> None:
        """Save the agent's state."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'replay_buffer': self.replay_buffer
        }, filepath)
        
    def load(self, filepath: str) -> None:
        """Load the agent's state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.replay_buffer = checkpoint['replay_buffer']
