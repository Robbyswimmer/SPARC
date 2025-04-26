"""
Curriculum Callback for SPARC agent.

This module provides a callback for updating the dataset curriculum based on agent performance.
"""
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import logging
from typing import Dict, Any, Optional, List
import wandb

from data.data_registry import CurriculumDataLoader

logger = logging.getLogger(__name__)

class CurriculumCallback(BaseCallback):
    """
    Callback for updating the dataset curriculum based on agent performance.
    
    This callback periodically checks QA scores and updates the curriculum
    by adding more challenging datasets when performance thresholds are met.
    
    Args:
        curriculum_loader: The curriculum data loader to update
        eval_freq: Update curriculum every `eval_freq` timesteps
        moving_avg_window: Window for calculating moving average QA score (episodes)
        verbose: Verbosity level (0: no output, 1: info)
    """
    
    def __init__(
        self,
        curriculum_loader: CurriculumDataLoader,
        eval_freq: int = 1000,
        moving_avg_window: int = 100,
        verbose: int = 0,
        update_fn = None  # Optional custom update function
    ):
        super().__init__(verbose)
        self.curriculum_loader = curriculum_loader
        self.eval_freq = eval_freq
        self.moving_avg_window = moving_avg_window
        self.update_fn = update_fn  # Store custom update function if provided
        
        # Track QA scores for moving average
        self.recent_qa_scores: List[float] = []
        self.current_avg_qa_score: float = 0.0
        self.n_calls_since_update = 0
        
    def _on_step(self) -> bool:
        """
        Update curriculum if it's time and QA score is above threshold.
        """
        self.n_calls_since_update += 1
        
        # Skip if not time to update yet
        if self.n_calls_since_update < self.eval_freq:
            return True
            
        # Reset counter
        self.n_calls_since_update = 0
        
        # Check if we have a recent QA score in the info dict
        info = self.locals.get("infos")
        if info and len(info) > 0:
            # Extract QA scores from completed episodes
            for ep_info in info:
                if "qa_score" in ep_info:
                    qa_score = ep_info["qa_score"]
                    
                    # Add to recent scores
                    self.recent_qa_scores.append(qa_score)
                    
                    # Keep only the most recent scores within window
                    if len(self.recent_qa_scores) > self.moving_avg_window:
                        self.recent_qa_scores.pop(0)
            
            # Calculate moving average
            if self.recent_qa_scores:
                self.current_avg_qa_score = np.mean(self.recent_qa_scores)
                
                # Try to update curriculum - use custom update function if provided
                if self.update_fn is not None:
                    updated = self.update_fn(self.current_avg_qa_score)
                else:
                    updated = self.curriculum_loader.update_curriculum(self.current_avg_qa_score)
                
                if updated and self.verbose > 0:
                    logger.info(f"Curriculum updated at step {self.num_timesteps}")
                    logger.info(f"Current QA Score: {self.current_avg_qa_score:.4f}")
                    logger.info(f"Active datasets: {self.curriculum_loader.active_datasets}")
                
                # Log curriculum metrics to wandb if available
                try:
                    if wandb.run is not None:
                        wandb.log({
                            "curriculum/avg_qa_score": self.current_avg_qa_score,
                            "curriculum/active_datasets": len(self.curriculum_loader.active_datasets),
                            "curriculum/level": self.curriculum_loader.current_level,
                            "global_step": self.num_timesteps
                        })
                except:
                    pass  # Ignore wandb errors
                    
        return True
