import gymnasium as gym
import numpy as np

print("Creating raw env...")
env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
print(
    "obs_space before wrappers:",
    env.observation_space,
    getattr(env.observation_space, "shape", None),
)
try:
    s = env.observation_space.sample()
    print("sample type:", type(s), "sample shape:", np.array(s).shape)
except Exception as e:
    print("sampling error:", e)
env.close()

print("\nCreating env with per-env wrappers...")
env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.ClipAction(env)
try:
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
    # FrameStack was renamed to FrameStackObservation in newer Gymnasium
    try:
        env = gym.wrappers.FrameStackObservation(env, 4)
    except AttributeError:
        # fallback for older versions
        env = gym.wrappers.FrameStack(env, 4)
    print(
        "obs_space after wrappers:",
        env.observation_space,
        getattr(env.observation_space, "shape", None),
    )
    obs, _ = env.reset(seed=0)
    print("reset obs type:", type(obs), "reset obs shape:", np.array(obs).shape)
except Exception as e:
    print("wrapping error:", e)
finally:
    env.close()
