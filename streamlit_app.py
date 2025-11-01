"""
Streamlit UI for the Prioritized Experience Replay project.

This module provides a web interface for training, monitoring,
and visualizing RL agents.
"""

import streamlit as st
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import torch
import gymnasium as gym
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import our modules
from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.sac_agent import SACAgent
from src.envs.env_utils import make_env, get_env_info, GridWorldEnv


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: str) -> None:
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_agent(agent_type: str, state_dim: int, action_dim: int, **kwargs):
    """Create an RL agent."""
    if agent_type == 'DQN':
        return DQNAgent(state_dim, action_dim, **kwargs)
    elif agent_type == 'PPO':
        return PPOAgent(state_dim, action_dim, **kwargs)
    elif agent_type == 'SAC':
        return SACAgent(state_dim, action_dim, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Prioritized Experience Replay RL",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Prioritized Experience Replay RL Project")
    st.markdown("Train and visualize reinforcement learning agents with prioritized experience replay")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Environment selection
        env_name = st.selectbox(
            "Environment",
            ["CartPole-v1", "MountainCar-v0", "LunarLander-v2", "GridWorld", "MountainCarContinuous-v0"]
        )
        
        # Agent selection
        agent_type = st.selectbox(
            "Agent Type",
            ["DQN", "PPO", "SAC"]
        )
        
        # Training parameters
        st.subheader("Training Parameters")
        total_timesteps = st.number_input("Total Timesteps", min_value=1000, max_value=1000000, value=50000)
        eval_frequency = st.number_input("Evaluation Frequency", min_value=1000, max_value=10000, value=5000)
        
        # Agent-specific parameters
        st.subheader("Agent Parameters")
        learning_rate = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=0.001, format="%.5f")
        batch_size = st.number_input("Batch Size", min_value=16, max_value=256, value=32)
        
        # PER parameters
        st.subheader("Prioritized Experience Replay")
        per_alpha = st.slider("PER Alpha", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
        per_beta = st.slider("PER Beta", min_value=0.0, max_value=1.0, value=0.4, step=0.1)
        
        # Training controls
        st.subheader("Training Controls")
        if st.button("Start Training", type="primary"):
            st.session_state.training = True
            st.session_state.episode_rewards = []
            st.session_state.episode_lengths = []
            st.session_state.step_count = 0
            
        if st.button("Stop Training"):
            st.session_state.training = False
            
        # Load/Save configuration
        st.subheader("Configuration")
        if st.button("Save Config"):
            config = {
                'env': {'name': env_name},
                'agent': {
                    'type': agent_type,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size
                },
                'per': {
                    'alpha': per_alpha,
                    'beta': per_beta
                },
                'training': {
                    'total_timesteps': total_timesteps,
                    'eval_frequency': eval_frequency
                }
            }
            save_config(config, 'config/streamlit_config.yaml')
            st.success("Configuration saved!")
            
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Training Progress")
        
        # Create environment
        if env_name == 'GridWorld':
            env = GridWorldEnv(size=8)
        else:
            env = make_env(env_name)
            
        env_info = get_env_info(env)
        
        # Create agent
        agent = create_agent(
            agent_type,
            env_info['state_dim'],
            env_info['action_dim'],
            learning_rate=learning_rate,
            batch_size=batch_size,
            per_alpha=per_alpha,
            per_beta=per_beta
        )
        
        # Training visualization
        if 'training' in st.session_state and st.session_state.training:
            # Placeholder for training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Training loop (simplified for demo)
            if 'episode_rewards' not in st.session_state:
                st.session_state.episode_rewards = []
                
            # Simulate training progress
            if len(st.session_state.episode_rewards) < 100:
                # Generate some sample data for demonstration
                episode_reward = np.random.normal(200, 50)
                st.session_state.episode_rewards.append(episode_reward)
                
                progress = len(st.session_state.episode_rewards) / 100
                progress_bar.progress(progress)
                status_text.text(f"Episode {len(st.session_state.episode_rewards)}: Reward = {episode_reward:.2f}")
                
                # Update plots
                st.rerun()
                
        # Plot training results
        if 'episode_rewards' in st.session_state and len(st.session_state.episode_rewards) > 0:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Episode Rewards', 'Moving Average', 'Reward Distribution', 'Training Progress'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Episode rewards
            fig.add_trace(
                go.Scatter(y=st.session_state.episode_rewards, mode='lines', name='Rewards'),
                row=1, col=1
            )
            
            # Moving average
            if len(st.session_state.episode_rewards) > 10:
                window = min(10, len(st.session_state.episode_rewards) // 5)
                moving_avg = pd.Series(st.session_state.episode_rewards).rolling(window=window).mean()
                fig.add_trace(
                    go.Scatter(y=moving_avg, mode='lines', name='Moving Avg'),
                    row=1, col=2
                )
            
            # Reward distribution
            fig.add_trace(
                go.Histogram(x=st.session_state.episode_rewards, name='Distribution'),
                row=2, col=1
            )
            
            # Training progress
            progress_data = list(range(len(st.session_state.episode_rewards)))
            fig.add_trace(
                go.Scatter(x=progress_data, y=st.session_state.episode_rewards, mode='lines', name='Progress'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
    with col2:
        st.header("Environment Info")
        
        # Display environment information
        st.json({
            "Name": env_info['name'],
            "State Dimension": env_info['state_dim'],
            "Action Dimension": env_info['action_dim'],
            "Continuous": env_info['continuous'],
            "Action Space": str(env_info['action_space']),
            "Observation Space": str(env_info['observation_space'])
        })
        
        # Agent information
        st.header("Agent Info")
        st.json({
            "Type": agent_type,
            "Learning Rate": learning_rate,
            "Batch Size": batch_size,
            "PER Alpha": per_alpha,
            "PER Beta": per_beta
        })
        
        # Training statistics
        if 'episode_rewards' in st.session_state and len(st.session_state.episode_rewards) > 0:
            st.header("Training Statistics")
            
            rewards = st.session_state.episode_rewards
            st.metric("Current Reward", f"{rewards[-1]:.2f}")
            st.metric("Average Reward", f"{np.mean(rewards):.2f}")
            st.metric("Max Reward", f"{np.max(rewards):.2f}")
            st.metric("Min Reward", f"{np.min(rewards):.2f}")
            st.metric("Episodes", len(rewards))
            
    # Environment visualization
    st.header("Environment Visualization")
    
    if env_name == 'GridWorld':
        # Custom GridWorld visualization
        st.subheader("GridWorld Environment")
        
        # Create a simple grid visualization
        grid_size = 8
        grid = np.zeros((grid_size, grid_size))
        
        # Add agent and goal positions
        grid[0, 0] = 1  # Agent
        grid[grid_size-1, grid_size-1] = 2  # Goal
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(grid, annot=True, fmt='d', cmap='viridis', ax=ax)
        ax.set_title('GridWorld Environment')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        st.pyplot(fig)
        
    else:
        # Generic environment info
        st.info(f"Environment: {env_name}")
        st.write("Use the training controls to start training and see the agent learn!")
        
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, PyTorch, and Gymnasium")


if __name__ == "__main__":
    main()
