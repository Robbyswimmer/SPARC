# src/train.py
import hydra, wandb
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from transformers import AutoTokenizer

from envs.streaming_qagym import StreamingQAGym
from agents.ppo_agent import TokenEmbedExtractor, WandbCallback
from agents.entropy_schedule import EntropyScheduleCallback
from agents.global_step_callback import GlobalStepCallback
from agents.eval_callback import ValidationCallback
from agents.curriculum_callback import CurriculumCallback
from data.data_registry import CurriculumDataLoader, mixed_stream, get_dataset_iterator
from curricula.length_schedule import LengthScheduleWrapper

import logging
logger = logging.getLogger(__name__)


class ChunkCurriculumCallback(BaseCallback):
    """
    Callback that advances chunk length curriculum counter based on actual training steps.
    
    This ensures the curriculum progresses properly in RL training, where the environment
    is only reset at episode completion rather than every training step.
    
    Args:
        curriculum_wrapper: The LengthScheduleWrapper instance to step forward
        frequency: How often to advance the curriculum counter (in timesteps)
        verbose: Verbosity level (0 for no output, 1 for info output, 2 for debug output)
    """
    def __init__(self, curriculum_wrapper: LengthScheduleWrapper, frequency: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.curriculum_wrapper = curriculum_wrapper
        self.frequency = frequency
        
    def _on_step(self) -> bool:
        # Advance the curriculum every [frequency] steps
        if self.n_calls % self.frequency == 0:
            self.curriculum_wrapper.step(1)
            # Log current curriculum level occasionally
            if self.n_calls % 1000 == 0:  # Report less frequently
                current_max = self.curriculum_wrapper.get_current_max_chunks()
                logger.info(f"Step {self.n_calls}: Chunk curriculum max chunks = {current_max}")
                # Log to wandb if available
                try:
                    if wandb.run is not None:
                        wandb.log({
                            "curriculum/max_chunks": current_max,
                            "global_step": self.num_timesteps
                        })
                except:
                    pass  # Ignore wandb errors
        return True

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer)

    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    # Initialize dataset curriculum settings
    dataset_enabled = cfg.data.get('curriculum_enabled', False)
    dataset_order = cfg.data.get('dataset_order', cfg.data.datasets)
    qa_thresholds = cfg.data.get('qa_thresholds', [0.3, 0.4, 0.5])
    
    # Setup either curriculum-based or standard dataset loading
    if dataset_enabled:
        logger.info(f"Initializing dataset curriculum with order: {dataset_order}")
        logger.info(f"QA thresholds for progression: {qa_thresholds}")
        
        # Create curriculum data loader with progressive dataset mixing
        curriculum_loader = CurriculumDataLoader(
            tokenizer=tokenizer,
            dataset_order=dataset_order,
            qa_thresholds=qa_thresholds,
            split=cfg.data.split,
            seed=cfg.train.seed,
            chunk_size=cfg.env.chunk_size,
            dataset_config=cfg.data.get('dataset_config', None)
        )
        
        # Get the iterator directly - not a function that returns an iterator
        # We'll wrap this in a lambda below to match expected interface
        train_iterator = curriculum_loader.current_iterator
    else:
        # Use standard mixed stream without curriculum
        logger.info(f"Using fixed mixed dataset stream with: {cfg.data.datasets}")
        
        # Create a mixed stream of all datasets
        def train_loader_fn():
            return mixed_stream(
                dataset_names=cfg.data.datasets,
                tokenizer=tokenizer,
                split=cfg.data.split,
                chunk_size=cfg.env.chunk_size,
                seed=cfg.train.seed
            )
    
    # Create a separate validation data loader function
    # This will be called each time validation runs to get fresh data
    def get_validation_data():
        logger.info("Creating validation data iterator")
        # For validation, use a mix of validation data from all datasets
        datasets = []
        valid_dataset_names = []
        
        # First try to load each dataset and track which ones succeed
        for name in cfg.data.datasets:
            # Get dataset-specific config if available
            dataset_config = {}
            
            # Apply dataset-specific configurations from the config file
            if hasattr(cfg.data, 'dataset_config') and hasattr(cfg.data.dataset_config, name):
                dataset_config = getattr(cfg.data.dataset_config, name)
                logger.info(f"Using custom config for {name}: {dataset_config}")
            
            # Special case for TriviaQA - if it keeps failing, allow skipping it
            if name == "triviaqa":
                # Get skip_validation setting, defaulting to False
                skip_validation = dataset_config.get('skip_validation', False) 
                if skip_validation:
                    logger.info(f"Skipping validation data for TriviaQA as configured")
                    continue
                
            try:
                # Use first 50 samples per dataset for validation
                iterator = get_dataset_iterator(
                    name, 
                    tokenizer, 
                    split="validation",
                    chunk_size=cfg.env.chunk_size,
                    **dataset_config  # Pass dataset-specific configs
                )
                # Test the iterator by getting one item
                try:
                    next(iterator)
                    # If we get here, the iterator works
                    datasets.append(iterator)
                    valid_dataset_names.append(name)
                    logger.info(f"Successfully loaded validation data for {name}")
                except Exception as e:
                    logger.warning(f"Failed to iterate validation data for {name}: {e}")
            except Exception as e:
                logger.warning(f"Failed to load validation data for {name}: {e}")
        
        if not datasets:
            # Fallback to using just NarrativeQA for validation if nothing else works
            logger.warning("No validation datasets could be loaded, falling back to NarrativeQA training data")
            try:
                iterator = get_dataset_iterator(
                    "narrativeqa", 
                    tokenizer, 
                    split="train",  # Use training data as fallback
                    chunk_size=cfg.env.chunk_size
                )
                datasets.append(iterator)
                valid_dataset_names.append("narrativeqa")
            except Exception as e:
                raise ValueError(f"Failed to load fallback validation data: {e}")
            
        # Create a mixed stream for validation using only the datasets that work
        logger.info(f"Using datasets for validation: {valid_dataset_names}")
        return mixed_stream(
            dataset_names=valid_dataset_names,  # Only use datasets that successfully loaded
            tokenizer=tokenizer,
            split="validation",
            seed=cfg.eval.seed
        )

    # 2) Create chunk length curriculum wrapper if requested
    chunk_curriculum_wrapper = None
    state_refs = {
        'chunk_curriculum_wrapper': None,
        'train_data_fn': None
    }
    if cfg.curriculum.enabled:
        logger.info(f"Using chunk length curriculum: {cfg.curriculum.start_chunks} → {cfg.curriculum.max_chunks} chunks")
        
        # If using dataset curriculum, we need to use the curriculum iterator as the base
        if dataset_enabled:
            # Create a wrapper around the curriculum iterator
            state_refs['chunk_curriculum_wrapper'] = LengthScheduleWrapper(
                base_iterator=train_iterator,  # Already an iterator, not a function
                initial_max_chunks=cfg.curriculum.start_chunks,
                final_max_chunks=cfg.curriculum.max_chunks,
                total_schedule_steps=cfg.curriculum.growth_steps
            )
            
            # Create a function that returns this wrapper (to match expected interface)
            # This function doesn't recreate the wrapper each time - it's already wrapped
            state_refs['train_data_fn'] = lambda: state_refs['chunk_curriculum_wrapper']
        else:
            # Standard approach without dataset curriculum - create a mixed stream
            mixed_iterator = mixed_stream(
                dataset_names=cfg.data.datasets,
                tokenizer=tokenizer,
                split=cfg.data.split,
                seed=cfg.train.seed
            )
            
            # Wrap the mixed stream in the chunk length curriculum
            state_refs['chunk_curriculum_wrapper'] = LengthScheduleWrapper(
                base_iterator=mixed_iterator,
                initial_max_chunks=cfg.curriculum.start_chunks,
                final_max_chunks=cfg.curriculum.max_chunks, 
                total_schedule_steps=cfg.curriculum.growth_steps,
            )
            
            # Create a function that returns the wrapped iterator
            state_refs['train_data_fn'] = lambda: state_refs['chunk_curriculum_wrapper']
    else:
        # No chunk length curriculum
        if dataset_enabled:
            # Use the curriculum iterator directly
            state_refs['train_data_fn'] = lambda: train_iterator
        else:
            # Create a standard mixed stream
            state_refs['train_data_fn'] = lambda: mixed_stream(
                dataset_names=cfg.data.datasets,
                tokenizer=tokenizer,
                split=cfg.data.split,
                seed=cfg.train.seed
            )

    # 3) Factory to create fresh envs that consume the loader
    def make_env(seed=None):
        def _thunk():
            env = StreamingQAGym(
                chunk_size   = cfg.env.chunk_size,
                max_window   = cfg.env.max_window,
                data_loader_fn = state_refs['train_data_fn'],
                gamma_step = cfg.env.get('gamma_step', 0.0), # Load gamma_step from config
                seed         = seed,
            )
            # If using wrapped loader_fn approach for dataset+chunk curriculum, 
            # store reference to the wrapper for the callback
            if dataset_enabled and cfg.curriculum.enabled:
                # Check if the data iterator is already a LengthScheduleWrapper
                if isinstance(env.ds_iter, LengthScheduleWrapper):
                    state_refs['chunk_curriculum_wrapper'] = env.ds_iter
                # Or check if it might be inside the iterator's __self__ attribute
                elif hasattr(env.ds_iter, '__self__') and hasattr(env.ds_iter.__self__, '__dict__'):
                    for item in env.ds_iter.__self__.__dict__.values():
                        if isinstance(item, LengthScheduleWrapper):
                            state_refs['chunk_curriculum_wrapper'] = item
                            break
            return env
        return _thunk

    vec_env = DummyVecEnv([make_env(seed=i) for i in range(cfg.train.n_envs)])
    vec_env = VecMonitor(vec_env)

    policy_kwargs = dict(
        features_extractor_class  = TokenEmbedExtractor,
        features_extractor_kwargs = dict(embed_dim=cfg.model.embed_dim,
                                         vocab_size=cfg.model.vocab_size),
        net_arch                  = [dict(pi=[128, 64], vf=[128, 64])],
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate = cfg.train.lr,
        n_steps       = cfg.train.n_steps,
        batch_size    = cfg.train.batch_size,
        gamma         = cfg.train.gamma,
        gae_lambda    = cfg.train.gae_lambda,
        clip_range    = cfg.train.clip_range,
        ent_coef      = 0.05,  # Start with high entropy for exploration
        policy_kwargs = policy_kwargs,
        tensorboard_log = "./runs/tb",
        device          = "auto",
    )

    run_name = f"{cfg.exp.name}-{wandb.util.generate_id()}"
    # Setup callbacks
    callbacks = [WandbCallback(project=cfg.wandb.project, run_name=run_name)]
    
    # Add chunk length curriculum callback if enabled
    if cfg.curriculum.enabled and state_refs['chunk_curriculum_wrapper'] is not None:
        chunk_callback = ChunkCurriculumCallback(
            curriculum_wrapper=state_refs['chunk_curriculum_wrapper'],
            frequency=1  # Advance every step
        )
        callbacks.append(chunk_callback)
        logger.info("Using chunk length curriculum progression callback")
    
    # Add validation callback for performance tracking on held-out data
    # Create this before dataset curriculum callback since it depends on it
    validation_callback = ValidationCallback(
        eval_freq=cfg.eval.frequency,      # How often to evaluate
        eval_episodes=cfg.eval.episodes,   # Number of validation episodes
        data_loader_fn=get_validation_data,  # Validation data loader function
        n_eval_envs=1,                     # Single environment for validation
        save_path=cfg.eval.save_path,      # Path to save best model
        name_prefix=cfg.eval.name_prefix,  # Prefix for best model file
        verbose=1
    )
    callbacks.append(validation_callback)
    print("Using validation callback for performance tracking and checkpointing")
    
    # Add dataset curriculum callback if enabled
    if dataset_enabled:
        
        # When curriculum loader is updated, we need to:
        # 1. Update the train_iterator in CurriculumDataLoader
        # 2. Update the train_data_fn to use the new iterator
        # 3. Possibly update chunk_curriculum_wrapper if using both curricula
        
        # Create a custom update function for the callback
        def update_dataset_curriculum(qa_score: float) -> bool:
            # Try to update curriculum
            updated = curriculum_loader.update_curriculum(qa_score)
            
            if updated:
                # If using chunk curriculum, need to rewrap the new iterator
                if cfg.curriculum.enabled and state_refs['chunk_curriculum_wrapper'] is not None:
                    # Create a new chunk curriculum wrapper with the updated iterator
                    state_refs['chunk_curriculum_wrapper'] = LengthScheduleWrapper(
                        base_iterator=curriculum_loader.current_iterator,
                        initial_max_chunks=cfg.curriculum.start_chunks,
                        final_max_chunks=cfg.curriculum.max_chunks,
                        total_schedule_steps=cfg.curriculum.growth_steps
                    )
                    
                    # Update the data function to use the new wrapper
                    state_refs['train_data_fn'] = lambda: state_refs['chunk_curriculum_wrapper']
                    
                    logger.info("Updated chunk curriculum wrapper with new dataset mix")
                else:
                    # Just update the training data function to use the new iterator
                    state_refs['train_data_fn'] = lambda: curriculum_loader.current_iterator
            
            return updated
            
        # Create and add the callback with our custom update function
        # Note: We pass the validation_callback to use its metrics instead of on-policy rollouts
        dataset_callback = CurriculumCallback(
            curriculum_loader=curriculum_loader,
            validation_callback=validation_callback,  # Use validation metrics for curriculum decisions
            eval_freq=cfg.eval.frequency,  # Check after each validation run
            verbose=1,
            update_fn=update_dataset_curriculum
        )
        callbacks.append(dataset_callback)
        print("Using dataset curriculum progression callback with validation metrics")
    
    # Add entropy schedule callback to linearly decay entropy coefficient
    entropy_callback = EntropyScheduleCallback(
        start_coef=0.05,
        end_coef=0.0,
        decay_fraction=0.5,  # Decay over first half of training
        total_timesteps=cfg.train.total_steps,
        verbose=1
    )
    callbacks.append(entropy_callback)
    print("Using entropy scheduling to control exploration")
    
    # Add global step callback for token reward annealing
    global_step_callback = GlobalStepCallback(verbose=1)
    callbacks.append(global_step_callback)
    print("Using global step tracking for token reward annealing")
    


    model.learn(
        total_timesteps = cfg.train.total_steps,
        callback        = callbacks
    )

    model.save(f"checkpoints/{run_name}.zip")

if __name__ == "__main__":
    main()
