"""
train_car.py - Proximal Policy Optimization (PPO) for CarRacing-v3

This script implements PPO to solve the Gymnasium CarRacing-v3 environment using pixel observations.
It is based on the CleanRL single-file implementation structure.

Observation Space: 
- 4 stacked grayscale frames, shape (4, 96, 96).

Architecture:
- 3 Conv2d layers.
- Dense layer (512 units).
- Actor head: Normal distribution mean over 3 continuous actions.
- Critic head: Scalar value estimate.

Reference: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
"""
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import shutil
import glob
import json

# detect whether moviepy (required by Gymnasium RecordVideo for saving) is available
try:
    import moviepy  # type: ignore

    MOVIEPY_AVAILABLE = True
except Exception:
    MOVIEPY_AVAILABLE = False

# torch.distributions.Normal is created inline when needed; no top-level import
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "CarRacing-v3"
    """the id of the environment"""
    total_timesteps: int = 250000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    eval_interval: int = 0
    """how many iterations between deterministic evaluation runs (0 = disabled)"""
    eval_episodes: int = 5
    """number of deterministic episodes to run during each evaluation"""
    eval_record_videos: bool = False
    """If True, record evaluation episodes to videos/<run_name>-eval via RecordVideo"""
    save_best_checkpoints: bool = False
    """If True, save a timestamped checkpoint when eval mean improves: best_model_iter{iteration}.pth"""
    min_action_std: float = 0.0
    """Minimum standard deviation for actions. Enforces exploration floor to avoid std collapse."""
    # retention and artifact options
    topk_eval_videos: int = 5
    """Keep top-K eval videos (ranked by return) per run; older lower-ranked videos removed."""
    topk_eval_checkpoints: int = 5
    """Keep top-K eval checkpoints (best_model_iter*.pth) per run; older lower-ranked ones removed."""
    topk_train_checkpoints: int = 5
    """Keep top-K training-best checkpoints (best_train_model_step*.pth) per run."""


def make_env(env_id, idx, capture_video, run_name, continuous: bool = False):
    """
    A corrected version of make_env for the image-based CarRacing environment.
    """

    def thunk():
        # The `continuous` flag must be passed in both cases.
        # If capture_video is requested for the first env, use rgb_array and record.
        # Otherwise, for the first env (idx==0) open a live window (render_mode='human')
        # so users running locally can see the environment. Non-zero envs keep default mode.
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", continuous=continuous)
            if MOVIEPY_AVAILABLE:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                print(
                    "moviepy not available: skipping RecordVideo wrapper. To record videos install 'moviepy' or 'gymnasium[other]'."
                )
        else:
            if idx == 0:
                # show a live window for the first environment when not recording
                try:
                    env = gym.make(env_id, render_mode="human", continuous=continuous)
                except TypeError:
                    # fallback for Gym versions that don't accept render_mode here
                    env = gym.make(env_id, continuous=continuous)
            else:
                env = gym.make(env_id, continuous=continuous)

        # These wrappers are fine and helpful for tracking performance.
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)

        # For image-based environments we should convert to grayscale and
        # stack frames per-environment (not on the vectorized wrapper).
        # FrameStackObservation returns an observation with shape
        # (frames, H, W, 1). Convert that to (H, W, frames) so the rest of
        # the code (which expects H,W,C) works unchanged.
        try:
            env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
            # FrameStack was renamed to FrameStackObservation in newer Gymnasium
            try:
                env = gym.wrappers.FrameStackObservation(env, 4)
            except AttributeError:
                env = gym.wrappers.FrameStack(env, 4)

            # Move the stacked frame axis into the channel axis and remove the
            # singleton channel dim: (frames, H, W, 1) -> (H, W, frames)
            def _merge_frames(obs):
                # obs may be np.ndarray or other array-like
                a = np.array(obs)
                # if last dim is 1, squeeze it
                if a.ndim == 4 and a.shape[-1] == 1:
                    a = a.squeeze(-1)
                # a is now (frames, H, W) -> move frames to the channel axis
                if a.ndim == 3:
                    a = np.moveaxis(a, 0, 2)
                return a

            # compute new observation space: old is (frames, H, W, 1) or similar
            old_shape = getattr(env.observation_space, "shape", None)
            if old_shape is not None and len(old_shape) >= 3:
                # Expecting (frames, H, W, 1) or (frames, H, W)
                frames = old_shape[0]
                height = old_shape[1]
                width = old_shape[2]
                new_shape = (height, width, frames)
                new_space = gym.spaces.Box(
                    low=0, high=255, shape=new_shape, dtype=env.observation_space.dtype
                )
            else:
                new_space = None

            env = gym.wrappers.TransformObservation(
                env, _merge_frames, observation_space=new_space
            )
        except Exception:
            # If the wrappers are not available / applicable, fall back to
            # returning the env unchanged so users can still run non-image envs.
            pass

        # DO NOT use FlattenObservation or NormalizeObservation for image-based learning.
        # The Agent's network handles the normalization (dividing by 255).
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# Replace the original Agent class with this CNN-based one.
class Agent(nn.Module):
    def __init__(self, envs, min_action_std: float = 0.0):
        super().__init__()
        # The shape of the stacked grayscale frames is (num_envs, 4, 96, 96)
        # PyTorch CNNs expect (N, C, H, W), so C=4, H=96, W=96.
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
        )
        # Initialize weights for stability
        for module in self.network:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layer_init(module)

        self.actor_mean = layer_init(
            nn.Linear(512, int(np.prod(envs.single_action_space.shape))), std=0.01
        )
        # we store a learnable logstd but enforce a minimum std at sampling time
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, int(np.prod(envs.single_action_space.shape)))
        )
        self.min_action_std = float(min_action_std)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_value(self, x):
        # Gymnasium gives (N, H, W, C), but PyTorch needs (N, C, H, W).
        # We also normalize pixel values from [0, 255] to [0.0, 1.0].
        return self.critic(self.network(x.permute(0, 3, 1, 2) / 255.0))

    def get_action_and_value(self, x, action=None):
        inp = x.permute(0, 3, 1, 2) / 255.0
        hidden = self.network(inp)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        # apply min std enforcement: compute std from logstd and ensure >= min_action_std
        action_std = torch.exp(action_logstd)
        if self.min_action_std and self.min_action_std > 0.0:
            # clamp on CPU numpy-safe tensor
            min_std_t = torch.tensor(
                self.min_action_std, device=action_std.device, dtype=action_std.dtype
            )
            action_std = torch.max(action_std, min_std_t)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(hidden),
        )

    # implement forward to satisfy nn.Module abstract requirement
    def forward(self, x):
        return self.get_action_and_value(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # ensure minibatch_size is at least 1
    if args.minibatch_size < 1:
        args.minibatch_size = 1
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # cudnn benchmarking improves performance for fixed-size inputs when not
    # using deterministic mode. Respect the user's deterministic flag.
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = not args.torch_deterministic

    # CUDA device info
    use_cuda = torch.cuda.is_available() and args.cuda
    if use_cuda:
        print(f"CUDA available. Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - {torch.cuda.get_device_name(i)} (capability: ", end="")
            try:
                print(torch.cuda.get_device_capability(i), end=")\n")
            except Exception:
                print("unknown)\n")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("CUDA not available or disabled. Running on CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # save run config for reproducibility and for replay tools
    try:
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        import json

        with open(f"runs/{run_name}/train_config.json", "w", encoding="utf-8") as fh:
            json.dump(vars(args), fh, indent=2)
        print(f"Saved run config to runs/{run_name}/train_config.json")
    except Exception:
        print("Failed to save run config; continuing without it.")

    # best eval tracking
    best_eval_return = -float("inf")
    # best training (on-policy) episodic return tracking
    best_train_return = -float("inf")
    best_eval_path = None

    # helper: prune top-K artifacts (eval videos & checkpoints, train checkpoints)
    def _prune_eval_checkpoints_and_videos(run_name_local: str):
        try:
            # collect eval checkpoints
            ckpt_pattern = os.path.join(
                f"runs/{run_name_local}", "best_model_iter*.pth"
            )
            ckpts = glob.glob(ckpt_pattern)
            ckpts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            # remove older checkpoints beyond topk
            if (
                args.topk_eval_checkpoints > 0
                and len(ckpts) > args.topk_eval_checkpoints
            ):
                for old in ckpts[args.topk_eval_checkpoints :]:
                    try:
                        os.remove(old)
                    except Exception:
                        pass

            # collect video metas under videos/{run_name_local}-eval-*
            video_meta = []
            for folder in glob.glob(os.path.join("videos", f"{run_name_local}-eval-*")):
                for meta in glob.glob(os.path.join(folder, "*.meta.json")):
                    try:
                        with open(meta, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        video_meta.append(
                            (meta, float(data.get("return", float("-inf"))))
                        )
                    except Exception:
                        pass
            # sort and prune
            video_meta.sort(key=lambda x: x[1], reverse=True)
            if args.topk_eval_videos > 0 and len(video_meta) > args.topk_eval_videos:
                for meta_path, _ in video_meta[args.topk_eval_videos :]:
                    try:
                        mp4 = os.path.splitext(meta_path)[0] + ".mp4"
                        os.remove(meta_path)
                        if os.path.exists(mp4):
                            os.remove(mp4)
                    except Exception:
                        pass
        except Exception:
            pass

    def _prune_train_checkpoints(run_name_local: str):
        try:
            pattern = os.path.join(
                f"runs/{run_name_local}", "best_train_model_step*.pth"
            )
            ckpts = glob.glob(pattern)
            ckpts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            if (
                args.topk_train_checkpoints > 0
                and len(ckpts) > args.topk_train_checkpoints
            ):
                for old in ckpts[args.topk_train_checkpoints :]:
                    try:
                        os.remove(old)
                    except Exception:
                        pass
        except Exception:
            pass

    # env setup
    # Use AsyncVectorEnv to run multiple environments in parallel processes.
    # This helps for CPU-bound environments (rendering / Box2D) and increases GPU utilization.
    env_fns = [
        make_env(args.env_id, i, args.capture_video, run_name, continuous=True)
        for i in range(args.num_envs)
    ]
    try:
        envs = gym.vector.AsyncVectorEnv(env_fns)
    except Exception:
        # Fallback to SyncVectorEnv if AsyncVectorEnv isn't available
        envs = gym.vector.SyncVectorEnv(env_fns)
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    agent = Agent(envs, min_action_std=args.min_action_std).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    # convert to tensor; if using CUDA, pin memory and do non_blocking transfer
    if device.type == "cuda":
        next_obs = torch.as_tensor(next_obs).pin_memory().to(device, non_blocking=True)
    else:
        next_obs = torch.as_tensor(next_obs)
    next_done = torch.zeros(args.num_envs).to(device)

    # monitoring: EMA for SPS and rolling episodic returns
    from collections import deque

    sps_ema = 0.0
    ema_alpha = 0.1
    recent_returns = deque(maxlen=100)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            # convert next_obs / next_done efficiently
            if device.type == "cuda":
                next_obs = (
                    torch.as_tensor(next_obs).pin_memory().to(device, non_blocking=True)
                )
                # convert boolean mask to float so math like `1.0 - next_done` works
                next_done = torch.as_tensor(next_done, dtype=torch.float32).to(device)
            else:
                next_obs = torch.as_tensor(next_obs)
                next_done = torch.as_tensor(next_done, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        r = info["episode"]["r"]
                        ep_len = info["episode"]["l"]
                        recent_returns.append(r)
                        avg_return = float(np.mean(recent_returns))
                        print(
                            f"global_step={global_step}, episodic_return={r}, avg_return(last{len(recent_returns)})={avg_return:.2f}"
                        )
                        writer.add_scalar("charts/episodic_return", r, global_step)
                        writer.add_scalar("charts/episodic_length", ep_len, global_step)
                        # rolling/online training metrics
                        writer.add_scalar(
                            "charts/rolling_return", avg_return, global_step
                        )

                        # optionally persist the model when training episodic return improves
                        if args.save_best_checkpoints and r > best_train_return:
                            best_train_return = r
                            try:
                                train_best_path = f"runs/{run_name}/best_train_model_step{global_step}.pth"
                                torch.save(agent.state_dict(), train_best_path)
                                # also keep a convenience copy named best_train_model.pth
                                torch.save(
                                    agent.state_dict(),
                                    f"runs/{run_name}/best_train_model.pth",
                                )
                                print(
                                    f"New best training episodic return {best_train_return:.3f}; saved training best model to {train_best_path}"
                                )
                            except Exception as e:
                                print(f"Failed to save training best model: {e}")
                            # prune older training-best checkpoints
                            try:
                                _prune_train_checkpoints(run_name)
                            except Exception:
                                pass

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # no debug checks in production

        # Optimizing the policy and value network
        # initialize these in case the inner loop doesn't run (static analyzers)
        old_approx_kl = torch.tensor(0.0)
        approx_kl = torch.tensor(0.0)
        pg_loss = torch.tensor(0.0)
        v_loss = torch.tensor(0.0)
        entropy_loss = torch.tensor(0.0)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    # use unbiased=False to avoid NaNs when minibatch size is 1
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std(unbiased=False) + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes and print summary
        elapsed = time.time() - start_time
        sps = int(global_step / elapsed) if elapsed > 0 else 0
        # EMA update for SPS
        if sps_ema == 0.0:
            sps_ema = float(sps)
        else:
            sps_ema = (1.0 - ema_alpha) * sps_ema + ema_alpha * float(sps)
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)

        # concise human-readable progress line with ETA
        avg_iter_time = elapsed / iteration if iteration > 0 else 0.0
        remaining_iters = max(0, args.num_iterations - iteration)
        eta_sec = remaining_iters * avg_iter_time
        eta = time.strftime("%H:%M:%S", time.gmtime(eta_sec))
        print(
            f"iter={iteration}/{args.num_iterations} | step={global_step} | SPS={sps} (ema={sps_ema:.1f}) | lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"pg_loss={pg_loss.item():.4f} | v_loss={v_loss.item():.4f} | entropy={entropy_loss.item():.4f} | expl_var={explained_var:.3f} | ETA={eta}"
        )

        # report GPU utilization estimate (best-effort using torch.cuda APIs)
        if use_cuda:
            try:
                gpu_mem = torch.cuda.memory_reserved() / (1024**2)
                print(
                    f"GPU mem reserved: {gpu_mem:.1f} MB | Device util: (check via nvidia-smi for precise numbers)"
                )
            except Exception:
                pass

        # Periodic deterministic evaluation and best-model checkpointing
        # Runs a small set of deterministic episodes (uses actor mean) every args.eval_interval iterations
        if args.eval_interval and iteration % args.eval_interval == 0:

            def run_deterministic_eval(
                env_id,
                episodes,
                Model,
                model_state_dict,
                device,
                record: bool = False,
                eval_run_name: str | None = None,
            ):
                """
                Run deterministic evaluation. If `record` is True, env will record videos
                into `videos/{eval_run_name}` via the existing `make_env` thunk.
                Returns (list_of_returns, list_of_video_paths)
                """
                eval_run_name = eval_run_name or f"{run_name}-eval-iter{iteration}"
                # Create a fresh single env for evaluation
                eval_env = make_env(env_id, 0, record, eval_run_name, continuous=True)()

                # Build agent and load weights
                class Dummy:
                    single_action_space = eval_env.action_space
                    single_observation_space = eval_env.observation_space

                eval_agent = Model(Dummy(), min_action_std=args.min_action_std).to(
                    device
                )
                try:
                    eval_agent.load_state_dict(model_state_dict)
                except Exception:
                    # fallback if state dict wrapped
                    if (
                        isinstance(model_state_dict, dict)
                        and "model_state_dict" in model_state_dict
                    ):
                        eval_agent.load_state_dict(model_state_dict["model_state_dict"])  # type: ignore
                    else:
                        eval_agent.load_state_dict(model_state_dict)

                eval_agent.eval()
                returns = []
                for ep in range(episodes):
                    obs_eval, _ = eval_env.reset()
                    done = False
                    total = 0.0
                    while True:
                        obs_t = (
                            torch.as_tensor(obs_eval, dtype=torch.float32)
                            .unsqueeze(0)
                            .to(device)
                        )
                        with torch.no_grad():
                            inp = obs_t.permute(0, 3, 1, 2) / 255.0
                            hidden = eval_agent.network(inp)
                            action_mean = eval_agent.actor_mean(hidden)
                            action = action_mean.cpu().numpy()[0]
                        result = eval_env.step(action)
                        if len(result) == 5:
                            obs_eval, reward, terminated, truncated, info = result
                            done = terminated or truncated
                        else:
                            obs_eval, reward, done, info = result  # type: ignore
                        total += (
                            float(np.array(reward).sum())
                            if isinstance(reward, (list, tuple, np.ndarray))
                            else float(reward)
                        )
                        if done:
                            returns.append(total)
                            break

                # locate videos if recording was enabled
                video_files = []
                if record:
                    videos_dir = os.path.join("videos", eval_run_name)
                    if os.path.isdir(videos_dir):
                        video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))
                        video_files.sort(key=os.path.getmtime, reverse=True)

                # write metadata for each video (so indexing can pick it up)
                meta_files = []
                try:
                    import json

                    for v, ret in zip(video_files, returns[-len(video_files) :]):
                        meta = {"return": float(ret), "iteration": int(iteration)}
                        meta_path = os.path.splitext(v)[0] + ".meta.json"
                        try:
                            with open(meta_path, "w", encoding="utf-8") as mh:
                                json.dump(meta, mh)
                            meta_files.append(meta_path)
                        except Exception:
                            pass
                except Exception:
                    meta_files = []

                eval_env.close()
                return returns, video_files

            # try to load current state dict and evaluate
            try:
                current_sd = agent.state_dict()
                eval_returns, eval_videos = run_deterministic_eval(
                    args.env_id,
                    args.eval_episodes,
                    Agent,
                    current_sd,
                    device,
                    record=bool(args.eval_record_videos),
                    eval_run_name=f"{run_name}-eval-iter{iteration}",
                )
                mean_eval = float(np.mean(eval_returns))
                std_eval = float(np.std(eval_returns))
                print(
                    f"Eval mean deterministic return (iter {iteration}): {mean_eval} std: {std_eval}"
                )
                writer.add_scalar("eval/mean_return", mean_eval, global_step)
                writer.add_scalar("eval/std_return", std_eval, global_step)
                # log per-episode eval returns
                for idx, val in enumerate(eval_returns):
                    writer.add_scalar(
                        "eval/episodic_return", float(val), global_step + idx
                    )

                # track and save best (timestamped) eval model and optionally record videos
                if mean_eval > best_eval_return:
                    best_eval_return = mean_eval
                    # timestamped best checkpoint
                    best_path_ts = f"runs/{run_name}/best_model_iter{iteration}.pth"
                    best_path = f"runs/{run_name}/best_model.pth"
                    try:
                        torch.save(agent.state_dict(), best_path_ts)
                        torch.save(agent.state_dict(), best_path)
                        print(
                            f"New best eval return {best_eval_return:.3f}; saved best model to {best_path_ts} and {best_path}"
                        )
                    except Exception as e:
                        print(f"Failed to save best eval model: {e}")

                    # if videos produced, print them and add to TB as text
                    if args.eval_record_videos and eval_videos:
                        if MOVIEPY_AVAILABLE:
                            print("Eval videos saved:")
                            for v in eval_videos:
                                print(" -", v)
                            writer.add_text(
                                "eval/videos", "\n".join(eval_videos), global_step
                            )
                        else:
                            # moviepy not available so videos were not actually written; inform user
                            print(
                                "Eval requested video recording but moviepy is not installed; no videos were saved."
                            )
                    # prune to keep only top-K eval artifacts
                    try:
                        _prune_eval_checkpoints_and_videos(run_name)
                    except Exception:
                        pass

            except Exception as e:
                print(f"Evaluation failed: {e}")

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        # Optional evaluation / upload using cleanrl_utils. If the package is not
        # available, skip these steps instead of crashing the script.
        try:
            from cleanrl_utils.evals.ppo_eval import evaluate  # type: ignore
        except Exception:
            evaluate = None
            print(
                "Optional package 'cleanrl_utils' not found or failed to import; skipping post-training evaluation.",
            )
        episodic_returns = None

        if evaluate is not None:
            try:
                episodic_returns = evaluate(
                    model_path,
                    make_env,
                    args.env_id,
                    eval_episodes=10,
                    run_name=f"{run_name}-eval",
                    Model=Agent,
                    device=device,
                    gamma=args.gamma,
                )
                for idx, episodic_return in enumerate(episodic_returns):
                    writer.add_scalar("eval/episodic_return", episodic_return, idx)
            except Exception:
                episodic_returns = None
                print("Evaluation via cleanrl_utils failed; skipping eval logging.")

        if args.upload_model:
            try:
                from cleanrl_utils.huggingface import push_to_hub  # type: ignore

                repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
                repo_id = (
                    f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
                )
                push_to_hub(
                    args,
                    episodic_returns,
                    repo_id,
                    "PPO",
                    f"runs/{run_name}",
                    f"videos/{run_name}-eval",
                )
            except Exception:
                print(
                    "Optional upload helper 'cleanrl_utils.huggingface' not available or failed; skipping upload.",
                )

    envs.close()
    writer.close()
