# src/train.py
import hydra, wandb, torch, gymnasium as gym
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from envs.streaming_qagym import StreamingQAGym
from agents.ppo_agent import TokenEmbedExtractor, WandbCallback
from data.narrativeqa import narrativeqa_loader
from data.hotpotqa    import hotpotqa_loader
from curricula.length_schedule import LengthCurriculum

DATASETS = {
    "narrativeqa": narrativeqa_loader,
    "hotpotqa"   : hotpotqa_loader,
    # add more loaders here
}

@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig):
    # 1) Build joint iterable over chosen datasets
    loaders = [DATASETS[name](split=cfg.data.split) for name in cfg.data.datasets]
    mixed_loader = itertools.cycle(itertools.chain(*loaders))  # simple round‑robin

    # 2) Wrap with curriculum if requested
    if cfg.curriculum.enabled:
        mixed_loader = LengthCurriculum(
            mixed_loader,
            start_chunks=cfg.curriculum.start_chunks,
            max_chunks=cfg.curriculum.max_chunks,
            growth_steps=cfg.curriculum.growth_steps,
        )

    # 3) Factory to create fresh envs that consume the loader
    def make_env(seed=None):
        def _thunk():
            return StreamingQAGym(
                chunk_size   = cfg.env.chunk_size,
                max_window   = cfg.env.max_window,
                data_iter    = mixed_loader,   # 🡐 pass iterator
                seed         = seed,
            )
        return _thunk

    vec_env = DummyVecEnv([make_env(seed=i) for i in range(cfg.train.n_envs)])
    vec_env = VecMonitor(vec_env)

    policy_kwargs = dict(
        features_extractor_class  = TokenEmbedExtractor,
        features_extractor_kwargs = dict(embed_dim=cfg.model.embed_dim,
                                         vocab_size=cfg.model.vocab_size),
        net_arch                  = cfg.model.net_arch,
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
        policy_kwargs = policy_kwargs,
        tensorboard_log = "./runs/tb",
        device          = "auto",
    )

    run_name = f"{cfg.exp.name}-{wandb.util.generate_id()}"
    model.learn(
        total_timesteps = cfg.train.total_steps,
        callback        = WandbCallback(project=cfg.wandb.project,
                                        run_name=run_name)
    )

    model.save(f"checkpoints/{run_name}.zip")

if __name__ == "__main__":
    main()
