# MobileNet-V2 Profile

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

## Contents
1. [Overview](#1-overview)
2. [Setup Instructions](#2-setup-instructions)
3. [Repository Overview](#3-repository-overview)
4. [Reproduction](#5-reproduction)
5. [Results](#5-results)

## 1. Overview

This repo contains the code for calculating the **Inference Times**, **Model Structures** and **Memory Reports** for the **MobileNet-V2** as a whole and individual layers as well as blocks.

I have performed experiments on **two** types of inputs:

`Size Format: (B,C,H,W)`

- **Cityscapes:** Input of size = (1,3,1024,2048)

- **PASCAL-VOC-2012:** Input of size = (1,3,500,334)

## 2. Setup Instructions

You can setup the repo by running the following commands:
```
$ git clone https://github.com/praeclarumjj3/BackBone-Profile.git
```

```
$ pip install -r requirements.txt
```

## 3. Repository Overview

The repository contains the following modules:

- `cpu/` - Scripts and stats for performance on a **CPU**.
- `model_structures/` - Contains **.txt** files with a description about the `MobilNet-V2 Layerwise Architecture`.
- `memory_reports/` - Contains **.txt** files with a description about the `MobilNet-V2 Parameter-wise Memory`.
- `mobilenetv2_stats/` - Contains **.txt** files with **inference times** for `MobilNet-V2`.
    - `/layers` - Contains **.txt** files with **layerwise inference times** for `MobilNet-V2`.
- `timer.py` - Python Script for calculating the **inference times**.
- `mem.py` - Python Script for calculating the **memory reports** and **model structures**.
- `stats.sh` - Shell Script for storing **inference times** for `MobilNet-V2`.
- `summary.sh` - Shell Script for storing **model structures** for `MobilNet-V2`.
- `mem_report.sh` - Shell Script for storing **memory reports** for `MobilNet-V2`.
- `requirements.txt` - Contains all the dependencies required for running the code.

## 4. Reproduction

### Inference Time

- To store inference time for `MobileNet-V2`, run the following command:

```
$ sh stats.sh
```

### Model Structure

- To store the structure for `MobileNet-V2`, run the following command:

```
$ sh summary.sh 
```

### Memory Report

- To store the memory reports for `MobileNet-V2`, run the following command:

```
$ sh mem_report.sh 
```


## 5. Results

All the experiments are performed with a `batch size=1` and 300 iterations.

### Inference Time (GPU)

- All the experiments are performed individually on a single **12 GB GeForce RTX 2080 Ti** and a single **12GB TitanXP**. 

- For `25 iterations` at the start, the period is considered as a [GPU-warm-up session](https://deci.ai/the-correct-way-to-measure-inference-time-of-deep-neural-networks/).

- The stats are from the files in the `mobilenetv2_stats/` folder.

- Calculated using [Event.record( )](https://pytorch.org/docs/stable/cuda.html#torch.cuda.Event) method in `torch.cuda`.

```
import torch

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
// Process
end.record()
print("Execution Time is {} ms".format(start.elapsed_time(end)))
```

### Memory Report

- The stats are from the files in `memory_reports/` folder.

- Calculated using the [pytorch-memlab](https://pypi.org/project/pytorch-memlab/) package.

#### Performance Stats

|       Model      |     Dataset        | Inference Time (ms) [2080Ti] | Inference Time (ms) [TitanXP]| Std (ms) [2080Ti] | Std (ms) [TitanXP]|   FPS [2080Ti] | FPS [TitanXP]  |  Allocated Memory (MB)  | # Params (M)  | # Tensors (M) | 
| ---------------- | ------------------ | ---------------------------- | ---------------------------- | ----------------- | ----------------- | -------------- | -------------- | ----------------------- | --------------| ------------- |
| **MobileNet-V2** |    *Cityscapes*    |       **33.627**             |       **54.314**             | *0.058*           | *0.150*           | **29.73**      | **18.41**      |   **37.58**             | **3.504**     |   **9.830**   |
| **MobileNet-V2** |   *PASCAL-VOC-12*  |       **5.570**              |       **5.795**              | *0.131*           | *0.144*           | **179.53**     | **172.56**     |   **14.49**             | **3.504**     |   **4.040**   |

### Inference Time (CPU) 

- All the experiments are performed on a single **Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz**.

- The stats are from the files in **Resnet-x's** corresponding `cpu/resnetx_stats/` folder; where `x ∈ {18, 34, 50, 101}`.

- Calculated using [time.perf_counter( )](https://docs.python.org/3/library/time.html#time.perf_counter).

```
import time

start = time.perf_counter()
// Process
end = time.perf_counter()

print("Execution Time is {} ms".format((end-start)*1000))
```


|       Model      |     Dataset        | Inference Time (ms) | Std (ms) |   FPS    | 
| ---------------- | ------------------ | ------------------- | -------- | -------- |
| **MobileNet-V2** |    *Cityscapes*    |       **560.022**   | *43.278* |  *1.78*  |   
| **MobileNet-V2** |   *PASCAL-VOC-12*  |       **70.180**    | *3.190*  |  *14.24* |    
