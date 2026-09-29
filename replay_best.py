"""Replay an explicit checkpoint with deterministic actions; optionally record video."""
import argparse
import torch
import numpy as np
from train_car import Agent, make_env  # uses the same make_env/Agent definitions


def make_single_env(
    env_id, render=False, capture_video=False, run_name="replay", continuous=True
):
    # reuse thunk from train_car; idx=0 and capture_video controls render mode
    thunk = make_env(env_id, 0, capture_video, run_name, continuous=continuous)
    env = thunk()
    return env


def build_agent_from_env(env, device):
    # Create a tiny helper object that mimics the vectorized env attributes Agent expects
    class Dummy:
        single_action_space = (
            env.single_action_space
            if hasattr(env, "single_action_space")
            else env.action_space
        )
        single_observation_space = (
            env.single_observation_space
            if hasattr(env, "single_observation_space")
            else env.observation_space
        )

    dummy = Dummy()
    agent = Agent(dummy).to(device)
    return agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--env-id", default="CarRacing-v3")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument(
        "--record", action="store_true", help="Record video to videos/<run_name>"
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    # create env (if record=True it will use RecordVideo inside make_env when capture_video True)
    env = make_single_env(
        args.env_id,
        render=(not args.record),
        capture_video=args.record,
        run_name="replay",
    )
    # If the above returned a VectorEnv-like object then adjust; expect a single env instance.
    # Build agent
    agent = build_agent_from_env(env, device)
    # Load model
    sd = torch.load(args.model_path, map_location=device)
    try:
        agent.load_state_dict(sd)
    except Exception:
        # if saved as dict with extra keys
        if "model_state_dict" in sd:
            agent.load_state_dict(sd["model_state_dict"])
        else:
            agent.load_state_dict(sd)

    agent.eval()

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while True:
            # Wrapped observations are uint8 (96, 96, 4).
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            # deterministic action: use actor mean (bypass sampling)
            with torch.no_grad():
                inp = obs_t.permute(0, 3, 1, 2) / 255.0
                hidden = agent.network(inp)
                action_mean = agent.actor_mean(hidden)
                action = action_mean.cpu().numpy()[0]
            # step (gymnasium returns obs, reward, terminated, truncated, info)
            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result  # fallback
            total_reward += (
                float(np.array(reward).sum())
                if isinstance(reward, (list, tuple, np.ndarray))
                else float(reward)
            )
            steps += 1
            if not args.record:
                # if rendering in human mode, the env will open a window (make_env uses render_mode='human' for idx==0)
                pass
            if done:
                print(f"Episode {ep + 1} reward={total_reward} steps={steps}")
                break

    env.close()


if __name__ == "__main__":
    main()
