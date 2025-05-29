"""
Callback to update global step counter in environments for reward annealing.
"""
from stable_baselines3.common.callbacks import BaseCallback
import wandb

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
        self._anneal_steps = None  # Cache for token_reward_anneal_steps
    
    def _get_base_env(self, env):
        """
        Recursively unwrap the environment to get the base environment.
        
        Args:
            env: The environment to unwrap
            
        Returns:
            The base environment (without wrappers)
        """
        if hasattr(env, "env"):
            return self._get_base_env(env.env)
        return env

    def _on_step(self) -> bool:
        """
        Update the global_step counter in all environments.
        
        Returns:
            True to continue training
        """
        # Try both SB3 VecEnv conventions
        envs = []
        if hasattr(self.training_env, "envs"):
            envs = self.training_env.envs
        elif hasattr(self.training_env, "venv") and hasattr(self.training_env.venv, "envs"):
            envs = self.training_env.venv.envs
        
        for e in envs:
            # Get the base environment by recursively unwrapping
            real_env = self._get_base_env(e)
            if hasattr(real_env, "set_global_step"):
                real_env.set_global_step(self.num_timesteps)
        
        # Get token_reward_anneal_steps from the first environment (only once)
        if self._anneal_steps is None and envs:
            # Get the first real env by recursively unwrapping
            base_env = self._get_base_env(envs[0])
            # Extract the parameter if the environment has it, else use default
            self._anneal_steps = getattr(base_env, "token_reward_anneal_steps", 50000)
            if self.verbose:
                print(f"[GlobalStep] Using token_reward_anneal_steps={self._anneal_steps} from environment")
        
        # Ensure we have a value even if no environments were found
        if self._anneal_steps is None:
            self._anneal_steps = 50000  # Default fallback value

        # Log the current anneal factor so you can chart it
        # Must match the gym's anneal logic:
        #   anneal = max(0.1, 1.0 - global_step/token_reward_anneal_steps)
        anneal = max(0.1, 1.0 - self.num_timesteps / self._anneal_steps)
        wandb.log({"anneal_coef": anneal, "global_step": self.num_timesteps, "token_reward_anneal_steps": self._anneal_steps})
        
        if self.verbose and self.num_timesteps % 1000 == 0:
            print(f"[GlobalStep] step → {self.num_timesteps}, anneal → {anneal:.3f}")
            
        return True
