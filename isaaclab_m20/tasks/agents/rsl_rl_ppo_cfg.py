"""rsl_rl PPO configuration for the M20 tasks.

Two runners:

* ``M20RslRlPpoRunnerCfg`` -- the static-standing task, unchanged.
* ``M20VelocityRslRlPpoRunnerCfg`` -- the velocity task, with the first formal
  training plan's hyper-parameters.

The stager that actually uses the velocity runner is ``scripts/train.py``: it
runs the four course stages back to back, loading the previous stage's
checkpoint each time, so ``max_iterations`` here is only the sum of the stage
budgets (300 + 700 + 1000 + 1250) used as the fallback total.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class M20RslRlPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "m20_wheeled"
    empirical_normalization = False
    # Keep actions inside [-1, 1] so the action scale (0.25 rad / 20 rad/s) is
    # the real command range; the runner default of 100 lets the policy ask for
    # ~100x the intended target amplitude.
    clip_actions = 1.0
    # Single observation group: both actor and critic consume "policy".
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class M20VelocityRslRlPpoRunnerCfg(M20RslRlPpoRunnerCfg):
    """The plan's PPO table, verbatim where it matters.

    Differences from the standing runner, and why:

    * hidden [256, 256] instead of [512, 256, 128] -- the plan's two-layer MLP.
      A deeper/narrower stack of the same parameter count is not obviously worse,
      but the plan fixes the architecture and comparing runs is only meaningful
      if the architecture does not also move.
    * lr 3e-4 instead of 1e-3 -- 1e-3 is a fine rate for a single static pose and
      a fast way to blow up a 6-stage course, where each stage starts from the
      previous stage's weights.
    * 8 mini-batches instead of 4 -- with 4096 envs x 24 steps = 98304 transitions
      per iteration, 8 batches is 12288 samples per update, which keeps each PPO
      update small enough for ``desired_kl = 0.01`` to actually bind.
    * entropy 0.005 instead of 0.01 -- halved with the lower learning rate; the
      standing task needs exploration around one pose, the velocity task needs to
      *keep* the gait it finds rather than keep jittering.
    * action std 0.5 rather than 1.0 -- the sampler puts the *same* sigma on all
      16 action dims, but the dims do not mean the same thing: one unit of leg
      action is 0.25 rad (14 deg), while one unit of wheel action is 8 rad/s
      (0.72 m/s at the contact). A std of 0.9 therefore commanded the wheels to
      +-7.2 rad/s (+-0.65 m/s) on every step of the "stand still" stage, which is
      far larger than the 0.15 m/s the standing terms are trying to hold -- the
      exploration noise alone recreated the wandering the reward was meant to
      remove. 0.5 still gives +-7 deg of leg jitter and +-0.36 m/s of wheel
      noise at the start, and rsl_rl shrinks the std as the policy improves.
    * save_interval 50 -- the stages are 300-1250 iterations long, so 100 would
      leave 7 checkpoints per stage and no way to bisect a stage that goes bad
      halfway.
    """

    experiment_name = "m20_velocity"
    # 300 + 700 + 1000 + 1250, see scripts/train.py
    max_iterations = 3250
    save_interval = 50

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        # rsl_rl normalises advantages and clips the gradient norm internally;
        # both are unconditional there, so there is nothing to set for
        # "Advantage Normalization: True". max_grad_norm is the plan's 1.0.
        max_grad_norm=1.0,
    )
