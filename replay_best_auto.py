# replay_best_auto.py
#!/usr/bin/env python3
"""replay_best_auto.py - find & replay latest best models with optional min-std enforcement

Find the latest best_model*.pth in the `runs/` directory, load it, and replay it.
This script finds the latest `best_model*.pth` under `runs/`, loads it, and replays it for a
small number of episodes. Use `--record` to have the first environment write video files.
"""

import argparse
import glob
import os
from datetime import datetime

import torch
import numpy as np

from train_car import Agent, make_env


def find_latest_model(runs_dir: str, pattern: str = "best_model*.pth") -> str:
    candidates = glob.glob(os.path.join(runs_dir, "**", pattern), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No model files matching {pattern} found under {runs_dir}")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def build_agent_from_env(env, device, min_action_std: float = 0.0):
    class Dummy:
        single_action_space = env.single_action_space if hasattr(env, "single_action_space") else env.action_space
        single_observation_space = env.single_observation_space if hasattr(env, "single_observation_space") else env.observation_space

    dummy = Dummy()
    agent = Agent(dummy, min_action_std=min_action_std).to(device)
    return agent


def replay_model(
    model_path: str,
    env_id: str = "CarRacing-v3",
    episodes: int = 3,
    deterministic: bool = True,
    device: str | None = None,
    record: bool = True,
    out_name: str | None = None,
    min_std: float = 0.0,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    base = out_name or f"replay-{os.path.splitext(os.path.basename(model_path))[0]}"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{base}-{timestamp}"

    thunk = make_env(env_id, 0, record, run_name, continuous=True)
    env = thunk()

    agent = build_agent_from_env(env, device, min_action_std=min_std)

    sd = torch.load(model_path, map_location=device)
    try:
        agent.load_state_dict(sd)
    except Exception:
        if isinstance(sd, dict) and "model_state_dict" in sd:
            agent.load_state_dict(sd["model_state_dict"])  # type: ignore
        else:
            agent.load_state_dict(sd)

    agent.eval()

    saved_videos = []
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                inp = obs_t.permute(0, 3, 1, 2) / 255.0
                hidden = agent.network(inp)
                action_mean = agent.actor_mean(hidden)
                if deterministic:
                    action_np = action_mean.cpu().numpy()[0]
                else:
                    action_logstd = agent.actor_logstd.expand_as(action_mean)
                    action_std = torch.exp(action_logstd)
                    if min_std and min_std > 0.0:
                        min_std_t = torch.tensor(min_std, device=action_std.device, dtype=action_std.dtype)
                        action_std = torch.max(action_std, min_std_t)
                    dist = torch.distributions.Normal(action_mean, action_std)
                    sampled = dist.sample()
                    action_np = sampled.cpu().numpy()[0]

            result = env.step(action_np)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result

            total_reward += float(np.array(reward).sum()) if isinstance(reward, (list, tuple, np.ndarray)) else float(reward)
            steps += 1
            if done:
                print(f"Episode {ep + 1}: reward={total_reward:.2f} steps={steps}")
                break

        videos_dir = os.path.join("videos", run_name)
        if os.path.isdir(videos_dir):
            vids = glob.glob(os.path.join(videos_dir, "*.mp4"))
            vids.sort(key=os.path.getmtime, reverse=True)
            saved_videos.extend(vids)

    env.close()
    return saved_videos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--pattern", default="best_model*.pth")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--stochastic", action="store_true", help="If set, sample actions instead of deterministic mean")
    p.add_argument("--device", default=None)
    p.add_argument("--record", action=argparse.BooleanOptionalAction, default=True, help="Record videos; --no-record opens a live window")
    p.add_argument("--out-name", default=None, help="Optional prefix for output video folder")
    p.add_argument("--min-std", type=float, default=0.0, help="Minimum action std to enforce when replaying")
    args = p.parse_args()

    deterministic = args.deterministic and not args.stochastic

    try:
        model_path = find_latest_model(args.runs_dir, args.pattern)
    except FileNotFoundError as exc:
        p.error(f"{exc}. Train with --eval-interval 10 first, or set --pattern to match your checkpoint.")
    print(f"Found model: {model_path}")

    vids = replay_model(
        model_path,
        episodes=args.episodes,
        deterministic=deterministic,
        device=args.device,
        record=args.record,
        out_name=args.out_name,
        min_std=args.min_std,
    )

    if vids:
        print("Saved videos:")
        for v in vids:
            print(" -", v)
    else:
        print("No videos were produced (RecordVideo may not have been triggered).")


if __name__ == "__main__":
    main()

