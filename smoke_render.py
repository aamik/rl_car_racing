from train_car import make_env
import time

thunk = make_env("CarRacing-v3", 0, False, "smoke_test", continuous=True)
env = thunk()
obs, _ = env.reset(seed=0)
for _ in range(10):
    a = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(a)
    time.sleep(0.02)
env.close()
print("smoke render done")
