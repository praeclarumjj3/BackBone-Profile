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

- To store inference time for different varaints of ResNet, run the following commands:

#### ResNet-18
```
$ sh stats18.sh 
```

#### ResNet-34
```
$ sh stats34.sh 
```

#### ResNet-50
```
$ sh stats50.sh 
```

#### ResNet-101
```
$ sh stats101.sh
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

All the experiments are performed on a single **12 GB GeForce RTX 2080 Ti** with a `batch size=1` and 300 iterations.

### Inference Time (GPU)

- All the experiments are performed on a single **12 GB GeForce RTX 2080 Ti**.

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


|     Model     |     Cityscapes     |  PASCAL-VOC-2012   |     BasicBlock     |     Bottleneck     | Inference Time (ms) | Std (ms) |    FPS    | 
| ------------- | ------------------ | ------------------ | ------------------ | ------------------ | --------------------| -------- | --------- |
| **ResNet-18** | :heavy_check_mark: |                    | :heavy_check_mark: |                    |       **40.211**    | *6.232*  | **24.86** |
| **ResNet-18** |                    | :heavy_check_mark: | :heavy_check_mark: |                    |       **4.693**     | *0.699*  | **213.08**|
| **ResNet-34** | :heavy_check_mark: |                    | :heavy_check_mark: |                    |       **67.246**    | *8.928*  | **14.87** |
| **ResNet-34** |                    | :heavy_check_mark: | :heavy_check_mark: |                    |       **6.964**     | *1.597*  | **143.59**|
| **ResNet-50** | :heavy_check_mark: |                    |                    | :heavy_check_mark: |       **130.713**   | *15.115* | **7.65**  |
| **ResNet-50** |                    | :heavy_check_mark: |                    | :heavy_check_mark: |       **14.146**    | *2.191*  | **70.69** |
| **ResNet-101**| :heavy_check_mark: |                    |                    | :heavy_check_mark: |       **251.154**   | *51.768* | **3.98**  |
| **ResNet-101**|                    | :heavy_check_mark: |                    | :heavy_check_mark: |       **35.928**    | *4.147*  | **27.83** |


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


|     Model     |     Cityscapes     |  PASCAL-VOC-2012   |     BasicBlock     |     Bottleneck     | Inference Time (ms) | Std (ms) |    FPS    | 
| ------------- | ------------------ | ------------------ | ------------------ | ------------------ | --------------------| -------- | --------- |
| **ResNet-18** | :heavy_check_mark: |                    | :heavy_check_mark: |                    |       **605.45**    | *36.14*  | **1.65**  |
| **ResNet-18** |                    | :heavy_check_mark: | :heavy_check_mark: |                    |       **60.75**     | *24.79*  | **16.46** |
| **ResNet-34** | :heavy_check_mark: |                    | :heavy_check_mark: |                    |       **------**    | *-----*  | **-----** |
| **ResNet-34** |                    | :heavy_check_mark: | :heavy_check_mark: |                    |       **------**    | *-----*  | **-----** |
| **ResNet-50** | :heavy_check_mark: |                    |                    | :heavy_check_mark: |       **------**    | *-----*  | **-----** |
| **ResNet-50** |                    | :heavy_check_mark: |                    | :heavy_check_mark: |       **------**    | *-----*  | **-----** |
| **ResNet-101**| :heavy_check_mark: |                    |                    | :heavy_check_mark: |       **------**    | *-----*  | **-----** |
| **ResNet-101**|                    | :heavy_check_mark: |                    | :heavy_check_mark: |       **------**    | *-----*  | **-----** |


### Memory Report

- The stats are from the files in `memory_reports/` folder.

- Calculated using the [pytorch-memlab](https://pypi.org/project/pytorch-memlab/) package.


|     Model     |     Cityscapes     |    PASCAL-VOC-2012    |     BasicBlock     |     Bottleneck     |  Allocated Memory (MB)  | # Params (M)  | # Tensors (M)  | 
| ------------- | ------------------ | --------------------- | ------------------ | ------------------ | ----------------------- | --------------| --------------|
| **ResNet-18** | :heavy_check_mark: |                       | :heavy_check_mark: |                    |   **68.69**             | **11.689**    |   **17.990**   | 
| **ResNet-18** |                    | :heavy_check_mark:    | :heavy_check_mark: |                    |   **46.60**             | **11.689**    |   **12.200**   |
| **ResNet-34** | :heavy_check_mark: |                       | :heavy_check_mark: |                    |   **108.16**            | **21.797**    |   **28.106**   |
| **ResNet-34** |                    | :heavy_check_mark:    | :heavy_check_mark: |                    |   **85.20**             | **21.797**    |   **22.315**   |
| **ResNet-50** | :heavy_check_mark: |                       |                    | :heavy_check_mark: |   **121.73**            | **25.557**    |   **31.901**   |
| **ResNet-50** |                    | :heavy_check_mark:    |                    | :heavy_check_mark: |   **100.23**            | **25.557**    |   **26.111**   |
| **ResNet-101**| :heavy_check_mark: |                       |                    | :heavy_check_mark: |   **194.65**            | **44.549**    |   **50.946**   |
| **ResNet-101**|                    | :heavy_check_mark:    |                    | :heavy_check_mark: |   **172.65**            | **44.549**    |   **45.155**   |
