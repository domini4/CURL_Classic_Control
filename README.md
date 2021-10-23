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

![Alt Text](https://github.com/domini4/CURL_Classic_Control/video/cart_10000.gif)
