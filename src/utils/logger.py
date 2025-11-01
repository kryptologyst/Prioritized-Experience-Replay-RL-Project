"""
Logging utilities for the RL project.

This module provides utilities for logging, experiment tracking,
and visualization.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import wandb
from torch.utils.tensorboard import SummaryWriter


def setup_logger(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger for the project.
    
    Args:
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger('rl_project')
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def setup_wandb(config: Dict[str, Any]) -> None:
    """
    Setup Weights & Biases logging.
    
    Args:
        config: Configuration dictionary
    """
    wandb_config = config.get('logging', {})
    
    if wandb_config.get('use_wandb', False):
        wandb.init(
            project=wandb_config.get('project', 'rl-prioritized-replay'),
            name=wandb_config.get('run_name', None),
            config=config,
            tags=wandb_config.get('tags', [])
        )


def setup_tensorboard(log_dir: Path) -> Optional[SummaryWriter]:
    """
    Setup TensorBoard logging.
    
    Args:
        log_dir: Directory for TensorBoard logs
        
    Returns:
        SummaryWriter instance or None
    """
    tb_dir = log_dir / 'tensorboard'
    tb_dir.mkdir(parents=True, exist_ok=True)
    
    return SummaryWriter(str(tb_dir))
