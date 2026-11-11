"""rsl_rl PPO configuration for the M20 tasks.

Two runners:

* ``M20RslRlPpoRunnerCfg`` -- the static-standing task, unchanged.
* ``M20VelocityRslRlPpoRunnerCfg`` -- the velocity task, with the first formal
  training plan's hyper-parameters.

The stager that actually uses the velocity runner is ``scripts/train.py``: it
runs the four course stages back to back, loading the previous stage's
checkpoint each time, so ``max_iterations`` here is only the sum of the stage
budgets (900 + 700 + 1200 + 1400) used as the fallback total.
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
    * entropy 0.001 instead of 0.01 -- a tenth of the plan's value. The entropy
      bonus is a constant push that *grows* log_std; it is only balanced by the
      gradient of the policy loss, which is tiny when the reward is capped (stage
      1 lives at ~0.05/step because the tracking kernels are already saturated).
      At 0.005 the actor drifted its std up with the reward flat, which is
      exploration paid for by nothing.
    * action std 0.3 rather than 1.0 -- the sampler puts the *same* sigma on all
      16 action dims, but the dims do not mean the same thing: one unit of leg
      action is 0.25 rad (14 deg), one unit of wheel action is 27.78 rad/s
      (2.5 m/s at the contact). A std of 0.9 commanded the wheels to +-25 rad/s
      (+-2.25 m/s) on every step of the "stand still" stage, which is more than the
      whole speed envelope; even 0.5 gives +-1.25 m/s, far above the 0.15 m/s the
      standing terms try to hold. With the 2.5 m/s envelope the compromise is
      0.3: +-8.3 rad/s of wheel noise (+-0.75 m/s, enough to break the standstill
      deadlock) and +-0.075 rad of leg noise (enough to find a gait), and rsl_rl
      shrinks the std as the policy improves.
    * save_interval 50 -- the stages are 700-1400 iterations long, so 100 would
      leave 7 checkpoints per stage and no way to bisect a stage that goes bad
      halfway.
    * ``schedule="adaptive"`` -- the plan's original setting, kept as the
      default. It writes a new learning rate into Adam's param groups every
      iteration (x1.5 / /1.5 on the measured KL), so the rate follows the
      measured progress instead of a fixed budget. The catch is that Adam's step
      is ~``lr * m_hat / sqrt(v_hat)``, i.e. very nearly *proportional to lr*, so
      the schedule also swings the step size: in ``m20_v5`` lr sat near 0 for
      stretches and then spiked to 1.5-2e-3, and the stage-1 policy fell over at
      iteration ~200 after 160 iterations of standing. Pass ``--fixed_lr`` on
      the command line to pin the rate at ``learning_rate`` instead; that is the
      more stable mode, and it also needs the longer stage budget (stage 1 at
      ~900 iterations rather than a few hundred, because 3e-4 is 5x slower than
      the rate adaptive had pushed the productive window to).
    """

    experiment_name = "m20_velocity"
    # 900 + 700 + 1200 + 1400, see scripts/train.py
    max_iterations = 4200
    save_interval = 50

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
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
        # 0.001 rather than the plan's 0.005: on stage 1 the reward is capped at
        # ~0.05/step (the tracking kernels are already saturated while the robot
        # holds still), so the policy-gradient term that shrinks log_std has
        # almost nothing to push against while the entropy bonus keeps paying.
        # The observed symptom was entropy creeping up with mean_reward flat --
        # i.e. the actor relaxing its std instead of improving. Fifth of the
        # plan's value, still non-zero so the std does not collapse early.
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        # Adaptive by default (the plan's setting); ``train.py --fixed_lr``
        # pins it at ``learning_rate``. See the class docstring for the trade.
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        # rsl_rl normalises advantages and clips the gradient norm internally;
        # both are unconditional there, so there is nothing to set for
        # "Advantage Normalization: True". max_grad_norm is the plan's 1.0.
        max_grad_norm=1.0,
    )
