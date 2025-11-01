"""
Main training script for the Prioritized Experience Replay project.

This script provides a unified interface for training different RL agents
with prioritized experience replay on various environments.
"""

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import gymnasium as gym
from tqdm import tqdm
import logging
from datetime import datetime

# Import our modules
from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.sac_agent import SACAgent
from src.envs.env_utils import make_env, get_env_info, GridWorldEnv
from src.utils.logger import setup_logger, setup_wandb, setup_tensorboard


class RLTrainer:
    """
    Main trainer class for RL agents.
    
    Args:
        config: Configuration dictionary
        log_dir: Directory for logging
    """
    
    def __init__(self, config: Dict[str, Any], log_dir: str = "logs"):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger(self.log_dir / "training.log")
        
        # Setup experiment tracking
        if config.get('logging', {}).get('use_wandb', False):
            setup_wandb(config)
        if config.get('logging', {}).get('use_tensorboard', True):
            self.tb_writer = setup_tensorboard(self.log_dir)
        else:
            self.tb_writer = None
            
        # Initialize environment
        self.env = self._create_environment()
        self.env_info = get_env_info(self.env)
        
        # Initialize agent
        self.agent = self._create_agent()
        
        # Training state
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.step_count = 0
        self.episode_count = 0
        
        self.logger.info(f"Initialized trainer with config: {config}")
        self.logger.info(f"Environment info: {self.env_info}")
        
    def _create_environment(self) -> gym.Env:
        """Create and configure the environment."""
        env_config = self.config.get('env', {})
        env_name = env_config.get('name', 'CartPole-v1')
        
        if env_name == 'GridWorld':
            return GridWorldEnv(size=env_config.get('size', 8))
        else:
            return make_env(
                env_name,
                render_mode=env_config.get('render_mode'),
                max_episode_steps=env_config.get('max_episode_steps')
            )
            
    def _create_agent(self):
        """Create the RL agent based on configuration."""
        agent_config = self.config.get('agent', {})
        agent_type = agent_config.get('type', 'DQN')
        
        # Common parameters
        common_params = {
            'state_dim': self.env_info['state_dim'],
            'action_dim': self.env_info['action_dim'],
            'device': self.config.get('device', 'auto'),
            **agent_config
        }
        
        if agent_type == 'DQN':
            return DQNAgent(**common_params)
        elif agent_type == 'PPO':
            return PPOAgent(**common_params)
        elif agent_type == 'SAC':
            return SACAgent(**common_params)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
    def train(self) -> None:
        """Main training loop."""
        training_config = self.config.get('training', {})
        total_timesteps = training_config.get('total_timesteps', 100000)
        eval_frequency = training_config.get('eval_frequency', 10000)
        save_frequency = training_config.get('save_frequency', 50000)
        
        self.logger.info(f"Starting training for {total_timesteps} timesteps")
        
        # Training loop
        pbar = tqdm(total=total_timesteps, desc="Training")
        
        while self.step_count < total_timesteps:
            episode_reward, episode_length = self._run_episode()
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_count += 1
            
            # Logging
            if self.episode_count % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                
                self.logger.info(
                    f"Episode {self.episode_count}, "
                    f"Avg Reward: {avg_reward:.2f}, "
                    f"Avg Length: {avg_length:.2f}, "
                    f"Steps: {self.step_count}"
                )
                
                if self.tb_writer:
                    self.tb_writer.add_scalar('Reward/Episode', episode_reward, self.episode_count)
                    self.tb_writer.add_scalar('Reward/Average', avg_reward, self.episode_count)
                    self.tb_writer.add_scalar('Length/Average', avg_length, self.episode_count)
                    
            # Evaluation
            if self.step_count % eval_frequency == 0:
                self._evaluate()
                
            # Save checkpoint
            if self.step_count % save_frequency == 0:
                self._save_checkpoint()
                
            pbar.update(self.step_count - pbar.n)
            
        pbar.close()
        self.logger.info("Training completed!")
        
    def _run_episode(self) -> tuple:
        """Run a single episode."""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        if isinstance(self.agent, PPOAgent):
            # PPO collects full episodes
            states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
            
        while True:
            if isinstance(self.agent, PPOAgent):
                action, log_prob, value = self.agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                dones.append(done)
                
                obs = next_obs
                episode_reward += reward
                episode_length += 1
                self.step_count += 1
                
                if done:
                    # Update PPO agent
                    if len(states) > 0:
                        metrics = self.agent.update(
                            np.array(states),
                            np.array(actions),
                            np.array(log_probs),
                            np.array(rewards),
                            np.array(values),
                            np.array(dones)
                        )
                        if metrics and self.tb_writer:
                            for key, value in metrics.items():
                                self.tb_writer.add_scalar(f'Loss/{key}', value, self.step_count)
                    break
                    
            else:
                # DQN/SAC style agents
                action = self.agent.select_action(obs)
                
                if isinstance(action, np.ndarray):
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                else:
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    
                done = terminated or truncated
                
                # Store experience
                self.agent.store_experience(obs, action, reward, next_obs, done)
                
                # Update agent
                loss = self.agent.update()
                if loss and self.tb_writer:
                    self.tb_writer.add_scalar('Loss/Training', loss, self.step_count)
                    
                obs = next_obs
                episode_reward += reward
                episode_length += 1
                self.step_count += 1
                
                if done:
                    break
                    
        return episode_reward, episode_length
        
    def _evaluate(self) -> None:
        """Evaluate the agent."""
        eval_episodes = 10
        eval_rewards = []
        
        for _ in range(eval_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            
            while True:
                action = self.agent.select_action(obs, training=False)
                
                if isinstance(action, np.ndarray):
                    obs, reward, terminated, truncated, _ = self.env.step(action)
                else:
                    obs, reward, terminated, truncated, _ = self.env.step(action)
                    
                episode_reward += reward
                
                if terminated or truncated:
                    break
                    
            eval_rewards.append(episode_reward)
            
        avg_eval_reward = np.mean(eval_rewards)
        std_eval_reward = np.std(eval_rewards)
        
        self.logger.info(f"Evaluation: {avg_eval_reward:.2f} ± {std_eval_reward:.2f}")
        
        if self.tb_writer:
            self.tb_writer.add_scalar('Reward/Evaluation', avg_eval_reward, self.step_count)
            self.tb_writer.add_scalar('Reward/Evaluation_Std', std_eval_reward, self.step_count)
            
    def _save_checkpoint(self) -> None:
        """Save agent checkpoint."""
        checkpoint_path = self.log_dir / f"checkpoint_{self.step_count}.pt"
        self.agent.save(str(checkpoint_path))
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
    def plot_results(self) -> None:
        """Plot training results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Moving average rewards
        window = min(100, len(self.episode_rewards) // 10)
        if window > 1:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title(f'Moving Average Rewards (window={window})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
            axes[0, 1].grid(True)
            
        # Episode lengths
        axes[1, 0].plot(self.episode_lengths)
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Length')
        axes[1, 0].grid(True)
        
        # Reward distribution
        axes[1, 1].hist(self.episode_rewards, bins=50, alpha=0.7)
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train RL agents with Prioritized Experience Replay')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for logging')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer
    trainer = RLTrainer(config, args.log_dir)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.agent.load(args.resume)
        trainer.logger.info(f"Resumed from checkpoint: {args.resume}")
        
    # Train
    trainer.train()
    
    # Plot results
    trainer.plot_results()
    
    # Save final model
    final_path = Path(args.log_dir) / 'final_model.pt'
    trainer.agent.save(str(final_path))
    trainer.logger.info(f"Saved final model: {final_path}")


if __name__ == '__main__':
    main()
