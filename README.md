# Evolving Simple Organisms

Three organisms learn to collect food and navigate obstacles through reinforcement learning.

## Experiment folders

| Folder | Experiment |
| --- | --- |
| `scripts/single_organism_food_learning` | Single-organism food-learning baseline and its Colab files. |
| `scripts/organism_rl` | Shared-policy multi-organism reinforcement-learning experiments. |

## Active experiment: PPO obstacle curriculum

The current experiment lives in `scripts/organism_rl`. One shared PPO policy trains three organisms in stages:

1. Collect food without obstacles.
2. Navigate around one obstacle.
3. Navigate around two fixed obstacles.

```bash
cd scripts/organism_rl
python ppo_curriculum_main.py --scenario scenarios/ppo_curriculum.json
```

Every 5,000 population episodes, the policy is evaluated on 100 fixed unseen environments. The next stage begins only after the configured blind-test target is reached. Results, checkpoints, and TensorBoard logs are written to `runs/ppo_obstacle_curriculum/<timestamp>/`.

```bash
tensorboard --logdir runs/ppo_obstacle_curriculum
```

The simulation uses pastel cells, cilia, a direction tail, yellow nutrients, and fixed gray obstacles. These visuals do not alter the learning dynamics.

## Staged food-navigation and evolution

For the evolutionary experiment, first train a reusable movement policy. Each episode has one cell with randomized body traits and a nutrient at a random visible direction. The run stops early only after the fixed blind evaluation reaches 90% food-reaching success.

```bash
cd scripts/organism_rl
env -u PYTHONPATH -u PYTHONHOME ../../.venv/bin/python food_navigation_pretrain.py
```

It writes `food_navigation_pretrained.pt` plus checkpoints at episodes 500, 2,500, and 5,000 under `runs/food_navigation_pretrain/<timestamp>/checkpoints/`.

Reuse that policy during evolution without updating it, so evolution cannot erase the learned movement skill:

```bash
env -u PYTHONPATH -u PYTHONHOME ../../.venv/bin/python evolutionary_ppo_main.py \
  --render \
  --pretrained-policy runs/food_navigation_pretrain/<timestamp>/checkpoints/food_navigation_pretrained.pt
```

Add `--train-policy` only for a later, explicit fine-tuning experiment.

Inspect a saved stage-1 policy without slowing training:

```bash
env -u PYTHONPATH -u PYTHONHOME ../../.venv/bin/python food_navigation_watch.py \
  runs/food_navigation_pretrain/<timestamp>/checkpoints/food_navigation_pretrained.pt
```
