import numpy as np
import torch
import gymnasium as gym
from train_car import make_env, Agent


def test_make_env_returns_correct_shape():
    thunk = make_env("CarRacing-v3", 0, False, "test", continuous=True)
    env = thunk()
    obs, _ = env.reset(seed=0)
    arr = np.array(obs)
    # Expect shape (H, W, C) after our TransformObservation
    assert arr.shape == (96, 96, 4)
    assert arr.dtype == np.uint8
    env.close()


def test_agent_forward_pass():
    thunk = make_env("CarRacing-v3", 0, False, "test", continuous=True)
    env = thunk()
    obs, _ = env.reset(seed=0)
    env.close()

    # Create a dummy env-like object with action_space
    class Dummy:
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    agent = Agent(Dummy())
    agent = agent.cpu()

    x = torch.tensor(np.expand_dims(np.array(obs), 0)).float()
    action, logprob, entropy, value = agent.get_action_and_value(x)
    assert action.shape[0] == 1
    assert value.shape[0] == 1
    assert not torch.isnan(action).any()
    assert not torch.isnan(value).any()


def test_make_env_capture_video(monkeypatch):
    # Avoid heavy moviepy dependency by monkeypatching RecordVideo to a no-op.
    import gymnasium as _gym

    monkeypatch.setattr(_gym.wrappers, "RecordVideo", lambda env, path: env)

    # ensure make_env can create an env with capture_video True for idx 0
    thunk = make_env("CarRacing-v3", 0, True, "test", continuous=True)
    env = thunk()
    # just check reset works and returns a correctly shaped observation
    obs, _ = env.reset(seed=1)
    arr = np.array(obs)
    assert arr.ndim == 3
    env.close()


def test_agent_get_value_shapes():
    # test that get_value returns a tensor shaped (N,1)
    thunk = make_env("CarRacing-v3", 0, False, "test", continuous=True)
    env = thunk()
    obs, _ = env.reset(seed=0)
    env.close()

    class Dummy:
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    agent = Agent(Dummy()).cpu()
    x = torch.tensor(np.expand_dims(np.array(obs), 0)).float()
    v = agent.get_value(x)
    assert v.shape[0] == 1
    assert v.shape[1] == 1


def test_preprocessing_errors_are_not_hidden(monkeypatch):
    def broken_grayscale(*args, **kwargs):
        raise RuntimeError("preprocessing failed")

    import pytest
    monkeypatch.setattr(gym.wrappers, "GrayscaleObservation", broken_grayscale)
    with pytest.raises(RuntimeError, match="preprocessing failed"):
        make_env("CarRacing-v3", 1, False, "test", continuous=True)()
