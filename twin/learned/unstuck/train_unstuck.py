"""Train the unstuck policy with PPO (RSL-RL 5.5) on the Isaac Lab env, headless, kit-less OVPhysX.

  python train_unstuck.py --num_envs 2048 --iters 800
Logs and checkpoints: E:/twin/learned/runs/unstuck/<run>/ (tensorboard + model_*.pt + metrics.csv)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from isaaclab.app import launch_simulation  # noqa: E402


def agent_cfg(iters: int, steps: int):
    from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=steps,
        max_iterations=iters,
        save_interval=50,
        experiment_name="unstuck",
        empirical_normalization=False,
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        clip_actions=1.0,
        actor=RslRlMLPModelCfg(
            hidden_dims=[256, 128, 64],
            activation="elu",
            obs_normalization=True,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.6),
        ),
        critic=RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.003,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=5.0e-4,
            schedule="adaptive",
            gamma=0.98,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--tiles", default=os.path.join(HERE, "data", "tiles_train.json"))
    ap.add_argument("--out", default="E:/twin/learned/runs/unstuck")
    ap.add_argument("--run", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--resume", default="")
    a = ap.parse_args()
    import unstuck_env as U

    cfg = U.UnstuckEnvCfg()
    cfg.scene.num_envs = a.num_envs
    cfg.tiles_path = a.tiles
    cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = 0.02, 5, 5
    cfg.seed = 7
    with launch_simulation(cfg, {"physics": "ovphysx"}):
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

        env = U.UnstuckEnv(cfg)
        agent = handle_deprecated_rsl_rl_cfg(agent_cfg(a.iters, a.steps), check_rsl_rl_version())
        log_dir = os.path.join(a.out, a.run)
        os.makedirs(log_dir, exist_ok=True)
        wenv = RslRlVecEnvWrapper(env, clip_actions=1.0)
        runner = create_rsl_rl_runner(wenv, agent, log_dir=log_dir)
        if a.resume:
            runner.load(a.resume)
        t0 = time.time()
        runner.learn(num_learning_iterations=a.iters, init_at_random_ep_len=True)
        print(f"Training time: {time.time() - t0:.0f} s; log dir {log_dir}")
        env.close()
