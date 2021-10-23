CURL Rainbow
=======
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

**Status**: License attached from paper authors

This is an implementation of [CURL: Contrastive Unsupervised Representations for
Reinforcement Learning](https://arxiv.org/abs/2004.04136) coupled with the [Data Efficient Rainbow method](https://arxiv.org/abs/1906.05243) for Classic Control Environments.

Run the following command with the game as an argument:

CartPole-v1
```
python main.py --T-max 100000 --game CartPole-v1 --V-max 200 --V-min -200 --max-episode-length 500
```

MountainCar-v0
```
python main.py --T-max 100000 --game MountainCar-v0 --V-max 20 --V-min -20 --max-episode-length 200
```

To install all dependencies, run `bash install.sh`. It is recommended that you install the dependencies manually based on the requirements document.

CartPole-v1 after 10000 steps

![cartpole after 10000 steps](video/cart_10000.gif)

CartPole-v1 after 50000 steps

![cartpole after 50000 steps](video/cart_50000.gif)

CartPole-v1 after 100000 steps

![cartpole after 100000 steps](video/cart_100000.gif)

MountainCar-v0 after 10000 steps

![mountaincar after 10000 steps](video/mount_10000.gif)

MountainCar-v0 after 50000 steps

![mountaincar after 50000 steps](video/mount_50000.gif)

MountainCar-v0 after 100000 steps

![mountaincar after 100000 steps](video/mount_100000.gif)