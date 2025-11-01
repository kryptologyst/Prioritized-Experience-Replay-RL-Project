"""
Prioritized Experience Replay Buffer Implementation

This module implements a prioritized experience replay buffer that samples transitions
based on their TD-error, allowing the agent to learn more efficiently from important
experiences.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import random


@dataclass
class Experience:
    """Represents a single experience tuple."""
    state: Union[np.ndarray, torch.Tensor]
    action: Union[int, np.ndarray]
    reward: float
    next_state: Union[np.ndarray, torch.Tensor]
    done: bool
    info: Optional[dict] = None


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Samples experiences based on their TD-error priority, with importance sampling
    correction to reduce bias introduced by non-uniform sampling.
    
    Args:
        capacity: Maximum number of experiences to store
        alpha: Prioritization strength (0 = uniform, 1 = full prioritization)
        beta: Importance sampling correction strength
        beta_increment: Amount to increment beta per step
        epsilon: Small constant to avoid zero priorities
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # Storage
        self.buffer: List[Experience] = []
        self.priorities: List[float] = []
        self.max_priority = 1.0
        
        # Position tracking
        self.position = 0
        self.size = 0
        
    def add(
        self, 
        state: Union[np.ndarray, torch.Tensor],
        action: Union[int, np.ndarray],
        reward: float,
        next_state: Union[np.ndarray, torch.Tensor],
        done: bool,
        td_error: Optional[float] = None,
        info: Optional[dict] = None
    ) -> None:
        """
        Add a new experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            td_error: TD-error for prioritization (if None, uses max priority)
            info: Additional info dict
        """
        experience = Experience(state, action, reward, next_state, done, info)
        
        # Calculate priority
        if td_error is not None:
            priority = abs(td_error) + self.epsilon
        else:
            priority = self.max_priority
            
        # Update max priority
        self.max_priority = max(self.max_priority, priority)
        
        # Add to buffer
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
            self.size += 1
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
            self.position = (self.position + 1) % self.capacity
            
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences based on their priorities.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (experiences, indices, importance_weights)
        """
        if self.size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} experiences from buffer of size {self.size}")
            
        # Calculate sampling probabilities
        priorities = np.array(self.priorities[:self.size])
        scaled_priorities = priorities ** self.alpha
        probabilities = scaled_priorities / scaled_priorities.sum()
        
        # Sample indices
        indices = np.random.choice(
            self.size, 
            size=batch_size, 
            replace=False, 
            p=probabilities
        )
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize weights
        
        return experiences, indices, weights
        
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities for given indices based on TD-errors.
        
        Args:
            indices: Buffer indices to update
            td_errors: New TD-errors for these experiences
        """
        for idx, td_error in zip(indices, td_errors):
            if idx < self.size:
                priority = abs(td_error) + self.epsilon
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
                
    def update_beta(self) -> None:
        """Increment beta for importance sampling correction."""
        self.beta = min(1.0, self.beta + self.beta_increment)
        
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
        
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough experiences for sampling."""
        return self.size >= batch_size
        
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.priorities.clear()
        self.position = 0
        self.size = 0
        self.max_priority = 1.0


class SumTree:
    """
    Sum Tree data structure for efficient prioritized sampling.
    
    This is an alternative implementation using a binary tree for O(log n) sampling
    instead of O(n) sampling with the standard implementation.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
        
    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
            
    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve sample index from tree."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
            
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
            
    def total(self) -> float:
        """Get total priority."""
        return self.tree[0]
        
    def add(self, p: float, data: Any) -> None:
        """Add experience with priority p."""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, p)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            
        if self.n_entries < self.capacity:
            self.n_entries += 1
            
    def update(self, idx: int, p: float) -> None:
        """Update priority at index."""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
        
    def get(self, s: float) -> Tuple[Any, int, float]:
        """Get experience, index, and priority."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return self.data[data_idx], idx, self.tree[idx]
