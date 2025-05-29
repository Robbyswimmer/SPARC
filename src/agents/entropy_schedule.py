"""
Entropy coefficient scheduler for PPO.

This module provides a callback to linearly decay the entropy coefficient
during training, which helps transition from exploration to exploitation.
"""
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, Dict, Any
import numpy as np
import wandb

class EntropyScheduleCallback(BaseCallback):
    """
    Callback for linearly decaying the entropy coefficient (ent_coef) in PPO.
    
    This helps control the exploration-exploitation trade-off during training.
    High entropy encourages exploration (diverse actions), while low entropy
    encourages exploitation (optimal actions).
    
    Args:
        start_coef: Initial entropy coefficient value
        end_coef: Final entropy coefficient value  
        decay_steps: Number of steps over which to decay the coefficient
        verbose: Verbosity level (0: no output, 1: info, 2: debug)
    """
    
    def __init__(
        self, 
        start_coef: float = 0.05,
        end_coef: float = 0.0,
        decay_fraction: float = 0.5,  # Decay over first half of training by default
        total_timesteps: Optional[int] = None,  # If provided, calculate decay_steps from this
        decay_steps: Optional[int] = None,  # Direct specification if preferred
        verbose: int = 0
    ):
        super().__init__(verbose=verbose)
        self.start_coef = start_coef
        self.end_coef = end_coef
        
        # Allow either direct decay_steps or calculate from total_timesteps
        if decay_steps is not None:
            self.decay_steps = decay_steps
        elif total_timesteps is not None:
            self.decay_steps = int(total_timesteps * decay_fraction)
        else:
            # Default fallback
            self.decay_steps = 50_000 # move this to config
        
        if self.verbose > 0:
            print(f"Initializing entropy schedule: {self.start_coef} → {self.end_coef} over {self.decay_steps} steps")
    
    def _on_step(self) -> bool:
        """
        Update the entropy coefficient based on current training step.
        
        Returns:
            True to continue training
        """
        # Calculate current step's entropy coefficient
        progress = min(1.0, self.num_timesteps / self.decay_steps)
        current_coef = self.start_coef - progress * (self.start_coef - self.end_coef)
        
        # Make sure we're using the actual model's ent_coef attribute
        # In case it wasn't initialized with our value
        if self.num_timesteps == 0 and hasattr(self.model, "ent_coef"):
            self.model.ent_coef = self.start_coef
        
        # Update model's entropy coefficient
        self.model.ent_coef = current_coef
        
        # Log entropy coefficient and policy entropy if available
        if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
            sb3_metrics = self.model.logger.name_to_value
            policy_entropy = sb3_metrics.get("train/entropy", None)
            
            # Log to wandb if it's imported and initialized
            try:
                if wandb.run is not None:
                    log_data = {
                        "entropy/coefficient": current_coef,
                        "global_step": self.num_timesteps
                    }
                    
                    if policy_entropy is not None:
                        log_data["entropy/policy"] = policy_entropy
                        
                    wandb.log(log_data)
            except (ImportError, AttributeError):
                if self.verbose > 1 and self.num_timesteps % 1000 == 0:
                    print("WandB not available for entropy logging")
            
            # Print verbose output
            if self.verbose > 0 and self.num_timesteps % 1000 == 0:
                print(f"Step {self.num_timesteps}: entropy_coef = {current_coef:.5f}")
                if policy_entropy is not None:
                    print(f"Current policy entropy: {policy_entropy:.5f}")
        
        return True
