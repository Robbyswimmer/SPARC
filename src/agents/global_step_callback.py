"""
Callback to update global step counter in environments for reward annealing.
"""
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional

class GlobalStepCallback(BaseCallback):
    """
    Update the global_step counter in environments to enable reward annealing.
    
    This callback passes the current training timestep to all environments in a vectorized
    environment, allowing for reward schedules that change based on training progress.
    
    Args:
        verbose: Verbosity level (0: no output, 1: info, 2: debug)
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
    
    def _on_step(self) -> bool:
        """
        Update the global_step counter in all environments.
        
        Returns:
            True to continue training
        """
        # Access the underlying environments through VecEnv
        envs = self.training_env.envs
        
        # Update global_step in each environment
        for env in envs:
            if hasattr(env, "set_global_step"):
                env.set_global_step(self.num_timesteps)
        
        if self.verbose > 0 and self.num_timesteps % 1000 == 0:
            print(f"Updated global_step to {self.num_timesteps} for token reward annealing")
            
        return True
