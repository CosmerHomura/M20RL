"""Train a PPO policy for the M20 wheeled quadruped."""
import argparse
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from m20rl.callbacks import RewardLogCallback, SaveAllCallback
from m20rl.config import EnvConfig, TrainConfig
from m20rl.env import M20Env


def make_env(rank: int, seed: int, reward_mode: str):
    def _thunk():
        env = M20Env(env_config=EnvConfig(reward_mode=reward_mode))
        env.reset(seed=seed)
        return Monitor(env)
    return _thunk


def build_vec_env(n_envs: int, seed: int, reward_mode: str):
    fns = [make_env(i, seed + i, reward_mode) for i in range(n_envs)]
    return DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-mode", choices=["full", "forward_only"],
                        default="forward_only")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--log-root", default="logs")
    args = parser.parse_args()

    # The policy MLP is tiny; extra BLAS threads only add synchronisation
    # overhead and compete with the environment processes.
    torch.set_num_threads(2)

    cfg = TrainConfig(seed=args.seed)
    if args.total_timesteps is not None:
        cfg.total_timesteps = args.total_timesteps
    if args.n_envs is not None:
        cfg.n_envs = args.n_envs
    if args.device is not None:
        cfg.device = args.device

    run_name = args.run_name or f"{args.reward_mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path(args.log_root) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    venv = build_vec_env(cfg.n_envs, cfg.seed, args.reward_mode)
    venv.seed(cfg.seed)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        learning_rate=cfg.learning_rate,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        policy_kwargs=cfg.policy_kwargs,
        tensorboard_log=str(log_dir / "tb"),
        device=cfg.device,
        seed=cfg.seed,
        verbose=1,
    )

    callbacks = [
        RewardLogCallback(),
        SaveAllCallback(save_freq=cfg.save_freq, save_path=str(log_dir / "checkpoints")),
    ]
    eval_env = None
    if not args.no_eval:
        eval_env = build_vec_env(1, cfg.seed + 1000, args.reward_mode)
        eval_env = VecNormalize(
            eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=True)
        callbacks.append(EvalCallback(
            eval_env,
            best_model_save_path=str(log_dir / "best_model"),
            log_path=str(log_dir / "eval"),
            eval_freq=max(cfg.eval_freq // cfg.n_envs, 1),
            n_eval_episodes=cfg.n_eval_episodes,
            deterministic=True,
            verbose=1,
        ))

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=False,
    )
    model.save(log_dir / "final_model")
    venv.save(str(log_dir / "final_vecnormalize.pkl"))
    print(f"saved model and normalisation to {log_dir}")

    venv.close()
    if eval_env is not None:
        eval_env.close()


if __name__ == "__main__":
    main()
