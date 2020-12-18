# PoseNet Controller

## About
This repository contains a game controller based on the PoseNet architecture. It allows you to control a sprite (or any arbitrary program) using just your outstretched arms. The core of this codebase is based on https://github.com/rwightman/posenet-python, without which this project would not be possible.

### Install

A suitable Python 3.x environment with a recent version of Tensorflow is required.

A conda environment setup as below should suffice: 
```
conda install tensorflow scipy pyyaml python=3.6
pip install opencv-python keyboard
```

### Usage

To start up the Posenet Controller, simply run:
```
$ python3 posenet_controller.py
```

The controls are as follows:
* Home position is both of your arms down
* Putting your left hand out triggers the left arrow key
* Putting your right hand out triggers the right arrow key
* Putting one or more hands up triggers the spacebar
