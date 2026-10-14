# evolving-simple-organisms
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yFYZgkbBtIFWZ6LVQ0an5vKQ654s3h-f?usp=sharing)

Colab: 

colab üzerinde çalıştırmak isteyenler için eğitim sonucunda gelen ağırlık dosyasını çalıştırabilir:

```
cd ~/evolving-simple-organisms/scripts/one-way-learning/colab

python test.py
```

start sminulation:

```
cd ~/evolving-simple-organisms/scripts/one-way-learning

python agent.py
```

new terminal:

`tensorboard --logdir=logs`

## REINFORCE experiments

The one-food baseline and multi-food scenario live in `scripts/REINFORCE`.

```bash
cd scripts/REINFORCE
python main.py --scenario scenarios/one_food.json
python main.py --scenario scenarios/multi_food.json
```

Each run writes checkpoints and TensorBoard events to `runs/<scenario>/<timestamp>/`.




for simulation photo:

![simulation](imgs/simulation.png)



update version:

![simulation_update](imgs/simulation-update.png)



Creating weights for an organism:

![simulation_update](imgs/Creating_weights_for_an_organism.png)
