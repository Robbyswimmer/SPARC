"""
Curriculum Callback for SPARC agent.

This module provides a callback for updating the dataset curriculum based on agent performance.
It uses validation metrics rather than on-policy rollout metrics to make curriculum decisions,
which prevents thrashing due to PPO noise.
"""
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import logging
import wandb

from data.data_registry import CurriculumDataLoader

logger = logging.getLogger(__name__)

class CurriculumCallback(BaseCallback):
    """
    Callback for updating the dataset curriculum based on agent performance.
    
    This callback periodically checks validation QA scores and updates the curriculum
    by adding more challenging datasets when performance thresholds are met.
    
    Using validation metrics instead of on-policy rollout metrics prevents thrashing
    due to PPO noise and provides a more stable curriculum progression.
    
    Args:
        curriculum_loader: The curriculum data loader to update
        validation_callback: The validation callback that provides metrics
        eval_freq: Update curriculum every `eval_freq` timesteps
        verbose: Verbosity level (0: no output, 1: info)
        update_fn: Optional custom update function
    """
    
    def __init__(
        self,
        curriculum_loader: CurriculumDataLoader,
        validation_callback = None,  # ValidationCallback that provides metrics
        eval_freq: int = 1000,
        verbose: int = 0,
        update_fn = None  # Optional custom update function
    ):
        super().__init__(verbose)
        self.curriculum_loader = curriculum_loader
        self.validation_callback = validation_callback
        self.eval_freq = eval_freq
        self.update_fn = update_fn  # Store custom update function if provided
        
        # Track validation QA scores
        self.current_val_qa_score: float = 0.0
        self.n_calls_since_update = 0
        self.last_update_step = 0
        
    def _on_step(self) -> bool:
        """
        Update curriculum if it's time and validation QA score is above threshold.
        
        This method checks if validation metrics are available and uses them to
        make curriculum progression decisions. This approach is more stable than
        using on-policy rollout metrics, which can be noisy due to PPO exploration.
        """
        # Debug: Print timestep info every 10 steps
        if self.verbose > 0 and self.n_calls % 10 == 0:
            print(f"Debug: CurriculumCallback - n_calls={self.n_calls}, num_timesteps={self.num_timesteps}, last_update_step={self.last_update_step}")
            if self.validation_callback is not None:
                print(f"Debug: ValidationCallback has metrics: {hasattr(self.validation_callback, 'last_metrics') and self.validation_callback.last_metrics is not None}")
                if hasattr(self.validation_callback, 'last_metrics') and self.validation_callback.last_metrics:
                    print(f"Debug: Validation metrics: {self.validation_callback.last_metrics}")
        
        # Only check for curriculum updates after validation runs
        if self.validation_callback is None or self.num_timesteps <= self.last_update_step:
            return True
            
        # Check if validation has run since our last update
        if not hasattr(self.validation_callback, 'last_metrics') or not self.validation_callback.last_metrics:
            return True
            
        # Get the latest validation QA score
        val_metrics = self.validation_callback.last_metrics
        if 'eval/mean_qa_score' in val_metrics:
            # Use validation QA score for curriculum decisions
            self.current_val_qa_score = val_metrics['eval/mean_qa_score']
            
            # Update the last update step to avoid checking again until next validation
            self.last_update_step = self.num_timesteps
                
            # Try to update curriculum - use custom update function if provided
            if self.update_fn is not None:
                updated = self.update_fn(self.current_val_qa_score)
            else:
                updated = self.curriculum_loader.update_curriculum(self.current_val_qa_score)
            
            if updated and self.verbose > 0:
                logger.info(f"Curriculum updated at step {self.num_timesteps}")
                logger.info(f"Current Validation QA Score: {self.current_val_qa_score:.4f}")
                logger.info(f"Active datasets: {self.curriculum_loader.active_datasets}")
                
            # Log curriculum metrics to wandb if available
            try:
                if wandb.run is not None:
                    wandb.log({
                        "curriculum/val_qa_score": self.current_val_qa_score,
                        "curriculum/active_datasets": len(self.curriculum_loader.active_datasets),
                        "curriculum/level": self.curriculum_loader.current_level,
                        "global_step": self.num_timesteps
                    })
            except:
                    pass  # Ignore wandb errors
                    
        return True
