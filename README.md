# ResNet Profile

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

## Contents
1. [Overview](#1-overview)
2. [Setup Instructions](#2-setup-instructions)
3. [Repository Overview](#3-repository-overview)
4. [Reproduction](#5-reproduction)
5. [Results](#5-results)

## 1. Overview

This repo contains the code for calculating the **Inference Time** for the different variants of **ResNet** as a whole and individual layers as well as blocks.

## 2. Setup Instructions

You can setup the repo by running the following commands:
```
$ git clone https://github.com/praeclarumjj3/ResNet-Profile.git

$ pip install -r requirements.txt
```

## 3. Repository Overview

The repository contains the following modules:

- `model_structures/` - Contains **.txt** files with a description about the `ResNet Layerwise Architecture`.
- `resnet18_stats/` - Contains **.txt** files with inference times for `ResNet-18`.
- `resnet34_stats/` - Contains **.txt** files with inference times for `ResNet-34`. 
- `resnet50_stats/` - Contains **.txt** files with inference times for `ResNet-50`.
- `timer.py` - Python Script for calculating the inference.
- `stats18.sh` - Shell Script for storing inference times for `ResNet-18`.
- `stats34.sh` - Shell Script for storing inference times for `ResNet-34`.
- `stats50.sh` - Shell Script for storing inference times for `ResNet-50`.
- `requirements.txt` - Contains all the dependencies required for running the code

## 4. Reproduction

- To store inference time for different varaints of ResNet, run the following commands:

### ResNet-18
```
$ sh stats18.sh 
```

### ResNet-34
```
$ sh stats34.sh 
```

### ResNet-50
```
$ sh stats50.sh 
```

- To store the structure for different varaints of ResNet, run the following command:

```
$ sh summary.sh 
```

## 5. Results

All the experiments are performed on a single **GeForce RTX 2080 Ti** with `batch_size=2`

| Model  | Dataset | BasicBlock  | Bottleneck | Inference Time |
| ------ | ------- | ----------- | ---------- | -------------- |

(Will be filled in Shortly)