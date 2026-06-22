import gymnasium as gym
import numpy as np
import torch
from train_car import make_env, Agent

# create env via thunk
thunk = make_env("CarRacing-v3", 0, False, "debug", continuous=True)
env = thunk()
obs, _ = env.reset(seed=0)
print(
    "obs dtype, shape, min/max:",
    obs.dtype,
    np.array(obs).shape,
    np.array(obs).min(),
    np.array(obs).max(),
)
# prepare batch dimension to simulate vectorized env
obs_batch = np.expand_dims(np.array(obs), 0)  # shape (1, H, W, C)
print("obs_batch shape:", obs_batch.shape)


# make dummy env-like object for Agent init
class DummyEnv:
    single_action_space = env.action_space


agent = Agent(DummyEnv())
# send to cpu for debug
agent = agent.cpu()

x = torch.tensor(obs_batch).float()
print("input tensor min/max:", x.min().item(), x.max().item())
# perform forward pass step by step
x_perm = x.permute(0, 3, 1, 2) / 255.0
print(
    "after permute min/max:",
    x_perm.min().item(),
    x_perm.max().item(),
    "shape",
    x_perm.shape,
)

with torch.no_grad():
    hidden = agent.network(x_perm)
    print(
        "hidden contains NaN?",
        torch.isnan(hidden).any().item(),
        "hidden shape",
        hidden.shape,
    )
    action_mean = agent.actor_mean(hidden)
    print(
        "action_mean:", action_mean, "any NaN?", torch.isnan(action_mean).any().item()
    )
    action_logstd = agent.actor_logstd.expand_as(action_mean)
    print("action_logstd:", action_logstd)

env.close()
