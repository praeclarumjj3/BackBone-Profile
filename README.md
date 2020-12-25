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

This repo contains the code for calculating the **Inference Times**, **Model Structures** and **Memory Reports** for the different variants of **ResNet** as a whole and individual layers as well as blocks.

I have performed experiments on **two** types of inputs:

`Size Format: (B,C,H,W)`

- **Cityscapes:** Input of size = (1,3,1024,2048)

- **Cityscapes:** Input of size = (1,3,500,334)

## 2. Setup Instructions

You can setup the repo by running the following commands:
```
$ git clone https://github.com/praeclarumjj3/ResNet-Profile.git
```

```
$ pip install -r requirements.txt
```

## 3. Repository Overview

The repository contains the following modules:

- `cpu/` - Scripts and stats for performance on a **CPU**.
- `model_structures/` - Contains **.txt** files with a description about the `ResNet Layerwise Architecture`.
- `memory_reports/` - Contains **.txt** files with a description about the `ResNet Parameter-wise Memory`.
- `resnet18_stats/` - Contains **.txt** files with **inference times** for `ResNet-18`.
    - `/layers` - Contains **.txt** files with **layerwise inference times** for `ResNet-18`.
- `resnet34_stats/` - Contains **.txt** files with **inference times** for `ResNet-34`.
    - `/layers` - Contains **.txt** files with **layerwise inference times** for `ResNet-34`.
- `resnet50_stats/` - Contains **.txt** files with **inference times** for `ResNet-50`.
    - `/layers` - Contains **.txt** files with **layerwise inference times** for `ResNet-50`.
- `resnet101_stats/` - Contains **.txt** files with **inference times** for `ResNet-101`.
    - `/layers` - Contains **.txt** files with **layerwise inference times** for `ResNet-50`.
- `timer.py` - Python Script for calculating the **inference times**.
- `mem.py` - Python Script for calculating the **memory reports** and **model structures**.
- `stats18.sh` - Shell Script for storing **inference times** for `ResNet-18`.
- `stats34.sh` - Shell Script for storing **inference times** for `ResNet-34`.
- `stats50.sh` - Shell Script for storing **inference times** for `ResNet-50`.
- `stats101.sh` - Shell Script for storing **inference times** for `ResNet-101`.
- `summary.sh` - Shell Script for storing **model structures** for all variants of `ResNet`.
- `mem_report.sh` - Shell Script for storing **memory reports** for all variants of `ResNet`.
- `requirements.txt` - Contains all the dependencies required for running the code.
- `perf_counter/` - Scripts for calculating performance using the **time.perf_counter()** method on a **GPU**. It gives different results than the **Event.record()** method with inference time being higher in the **former one (perf_counter)**.

## 4. Reproduction

### Inference Time

- To store inference time for different varaints of ResNet, run the following command:

```
$ sh stats[x].sh // where x=18/34/50/101
```

### Model Structure

- To store the structure for different varaints of `ResNet`, run the following command:

```
$ sh summary.sh 
```

### Memory Report

- To store the memory reports for different varaints of `ResNet`, run the following command:

```
$ sh mem_report.sh 
```


## 5. Results

All the experiments are performed with a `batch size=1` and 300 iterations.

### Inference Time (GPU)

- All the experiments are performed individually on a single **12 GB GeForce RTX 2080 Ti** and a single **12GB TitanXP**. 

- For `25 iterations` at the start, the period is considered as a [GPU-warm-up session](https://deci.ai/the-correct-way-to-measure-inference-time-of-deep-neural-networks/).

- The stats are from the files in **Resnet-x's** corresponding `resnetx_stats/` folder; where `x ∈ {18, 34, 50, 101}`.

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

#### Performance on Cityscapes

|     Model     |     BasicBlock     |     Bottleneck     | Inference Time (ms) [2080Ti] | Inference Time (ms) [TitanXP]| Std (ms) [2080Ti] | Std (ms) [TitanXP]|   FPS [2080Ti] | FPS [TitanXP]  |  Allocated Memory (MB)  | # Params (M)  | # Tensors (M)  | 
| ------------- | ------------------ | ------------------ | ---------------------------- | ---------------------------- | ----------------- | ----------------- | -------------- | -------------- | ----------------------- | --------------| -------------- |
| **ResNet-18** | :heavy_check_mark: |                    |       **18.719**             |       **23.622**             | *0.104*           | *0.199*           | **53.42**      | **42.33**      |   **68.69**             | **11.689**    |   **17.990**   |
| **ResNet-34** | :heavy_check_mark: |                    |       **31.779**             |       **38.588**             | *0.228*           | *0.595*           | **31.46**      | **25.91**      |   **108.16**            | **21.797**    |   **28.106**   |
| **ResNet-50** |                    | :heavy_check_mark: |       **61.397**             |       **82.334**             | *0.386*           | *0.792*           | **16.28**      | **12.14**      |   **121.73**            | **25.557**    |   **31.901**   |
| **ResNet-101**|                    | :heavy_check_mark: |       **100.426**            |       **122.491**            | *1.184*           | *1.356*           | **9.95**       | **8.16**       |   **194.65**            | **44.549**    |   **50.946**   |

#### Performance on PASCAL-VOC-2012

|     Model     |     BasicBlock     |     Bottleneck     | Inference Time (ms) [2080Ti] | Inference Time (ms) [TitanXP]| Std (ms) [2080Ti] | Std (ms) [TitanXP]|   FPS [2080Ti] | FPS [TitanXP]  |  Allocated Memory (MB)  | # Params (M)  | # Tensors (M)  | 
| ------------- | ------------------ | ------------------ | ---------------------------- | ---------------------------- | ----------------- | ----------------- | -------------- | -------------- | ----------------------- | --------------| -------------- |
| **ResNet-18** | :heavy_check_mark: |                    |       **2.547**              |       **2.940**              | *0.213*           | *0.203*           | **392.61**     | **340.13**     |   **46.60**             | **11.689**    |   **12.200**   |
| **ResNet-34** | :heavy_check_mark: |                    |       **5.197**              |       **4.959**              | *0.186*           | *0.238*           | **192.41**     | **201.65**     |   **85.20**             | **21.797**    |   **22.315**   |
| **ResNet-50** |                    | :heavy_check_mark: |       **7.628**              |       **8.927**              | *0.232*           | *0.241*           | **131.09**     | **112.01**     |   **100.23**            | **25.557**    |   **26.111**   |
| **ResNet-101**|                    | :heavy_check_mark: |       **12.579**             |       **14.509**             | *0.428*           | *0.013*           | **79.49**      | **68.92**      |   **172.65**            | **44.549**    |   **45.155**   |

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

#### Performance on Cityscapes

|     Model     |     BasicBlock     |     Bottleneck     | Inference Time (ms) | Std (ms) |    FPS    | 
| ------------- | ------------------ | ------------------ | --------------------| -------- | --------- |
| **ResNet-18** | :heavy_check_mark: |                    |       **566.75**    | *35.50*  | **1.76**  |
| **ResNet-34** | :heavy_check_mark: |                    |       **807.57**    | *20.213* | **1.23**  |
| **ResNet-50** |                    | :heavy_check_mark: |       **1626.05**   | *92.88*  | **0.61**  |
| **ResNet-101**|                    | :heavy_check_mark: |       **2344.98**   | *120.06* | **0.42**  |

#### Performance on PASCAL-VOC-2012

|     Model     |     BasicBlock     |     Bottleneck     | Inference Time (ms) | Std (ms) |    FPS    | 
| ------------- | ------------------ | ------------------ | --------------------| -------- | --------- |
| **ResNet-18** | :heavy_check_mark: |                    |       **55.79**     | *3.08*   | **17.92** |
| **ResNet-34** | :heavy_check_mark: |                    |       **78.77**     | *5.37*   | **12.69** |
| **ResNet-50** |                    | :heavy_check_mark: |       **133.71**    | *7.469*  | **7.47**  |
| **ResNet-101**|                    | :heavy_check_mark: |       **223.59**    | *13.51*  | **4.47**  |
