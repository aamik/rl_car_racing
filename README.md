# RL Car Racing

A pixel-based PPO agent for Gymnasium's continuous **CarRacing-v3** environment. The project brings together a convolutional actor-critic, local TensorBoard tracking, checkpoint replay and video inspection in a small, script-based workflow adapted from CleanRL.

![CarRacing environment preview](assets/pygame.png)

This is a learning and experimentation project, not a claim that the environment is solved. Screenshots illustrate the workflow; pretrained checkpoints and a benchmark evaluation are not included.

## Run locally

Use Python 3.12+ and [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/aamik/rl_car_racing.git
cd rl_car_racing
uv sync --frozen
uv run python -m pytest
```

CarRacing uses Box2D and Pygame. The first environment opens a window when video recording is off. On a machine without a display, use `SDL_VIDEODRIVER=dummy` for tests. Building Box2D requires a C/C++ toolchain. The uv configuration supplies SWIG in its isolated build environment. The locked PyTorch distribution can also download several GB of CUDA libraries on Linux, even when running CPU tests.

## Training and inspection

```bash
uv run python train_car.py --total-timesteps 250000 --eval-interval 10 --save-best-checkpoints
uv run tensorboard --logdir runs
```

The training budget is an example, not a guaranteed performance target. CUDA is selected when available; `--no-cuda` uses the CPU. Evaluation runs every ten training iterations in this example. Use `--help` for the remaining PPO and artifact-retention options.

![Local TensorBoard training metrics](assets/tensorboard.png)

After training, replay the most recently modified evaluation-best checkpoint and record video:

```bash
uv run python replay_best_auto.py --episodes 3
uv run python video_index.py
```

Open `videos/index.html` to browse recordings. `replay_best_auto.py --no-record` opens a live window instead. A fresh clone has no checkpoints: train first, or supply a compatible checkpoint explicitly:

```bash
uv run python replay_best.py --model-path runs/YOUR_RUN/best_model.pth --record
```

The automatic selector uses modification time, not a comparison of scores across experiments. Training-best checkpoints can be selected with `--pattern 'best_train_model*.pth'`.

## Implementation

- **Observations:** grayscale conversion and four-frame stacking; the environment returns `(96, 96, 4)` uint8 arrays, converted to `(N, 4, 96, 96)` and scaled inside the CNN path.
- **Policy:** three convolutional layers, a 512-unit dense layer, Gaussian action means for steering/gas/brake and a scalar value head. Actions are clipped to environment bounds.
- **Learning:** PPO with generalized advantage estimation, clipped updates and optional learning-rate annealing.
- **Artifacts:** local metrics and checkpoints in `runs/`, recordings in `videos/`. These generated files are ignored by Git. External tracking is opt-in through `--track` and the `tracking` extra.

## Files

| File | Purpose |
| --- | --- |
| `train_car.py` | Environment wrappers, actor-critic and PPO training loop |
| `replay_best.py` | Replay an explicit checkpoint |
| `replay_best_auto.py` | Find and replay the latest matching checkpoint |
| `video_index.py` | Build a browsable index, including optional return metadata |
| `inspect_env.py`, `debug_agent.py`, `smoke_render.py` | Manual environment, tensor and rendering diagnostics |
| `tests/` | Environment shape and policy-forward checks |

The single-file PPO structure is adapted from CleanRL, as documented in `train_car.py`. Seeds and run settings support repeatable experiments, but results may vary with hardware and library versions. A stronger evaluation would compare multiple training seeds on a fixed, separate set of tracks.
