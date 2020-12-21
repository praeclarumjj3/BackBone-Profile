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

- `model_structures/` - Contains **.txt** files with a description about the `ResNet Layerwise Architecture`.
- `memory_reports/` - Contains **.txt** files with a description about the `ResNet Parameter-wise Memory`.
- `resnet18_stats/` - Contains **.txt** files with **inference times** for `ResNet-18`.
- `resnet34_stats/` - Contains **.txt** files with **inference times** for `ResNet-34`. 
- `resnet50_stats/` - Contains **.txt** files with **inference times** for `ResNet-50`.
- `resnet101_stats/` - Contains **.txt** files with **inference times** for `ResNet-101`.
- `timer.py` - Python Script for calculating the **inference times**.
- `mem.py` - Python Script for calculating the **memory reports** and **model structures**.
- `stats18.sh` - Shell Script for storing **inference times** for `ResNet-18`.
- `stats34.sh` - Shell Script for storing **inference times** for `ResNet-34`.
- `stats50.sh` - Shell Script for storing **inference times** for `ResNet-50`.
- `stats101.sh` - Shell Script for storing **inference times** for `ResNet-101`.
- `summary.sh` - Shell Script for storing **model structures** for all variants of `ResNet`.
- `mem_report.sh` - Shell Script for storing **memory reports** for all variants of `ResNet`.
- `requirements.txt` - Contains all the dependencies required for running the code

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

All the experiments are performed on a single **12 GB GeForce RTX 2080 Ti** with a single image as input.

### Inference Time

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


|     Model     |     Cityscapes     |  PASCAL-VOC-2012   |     BasicBlock     |     Bottleneck     |   Inference Time (ms)   |    FPS    |
| ------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------------|-----------|
| **ResNet-18** | :heavy_check_mark: |                    | :heavy_check_mark: |                    |         **103.44**      | **9.66**  |
| **ResNet-18** |                    | :heavy_check_mark: | :heavy_check_mark: |                    |         **10.40**       | **96.15** |
| **ResNet-34** | :heavy_check_mark: |                    | :heavy_check_mark: |                    |         **173.41**      | **5.76**  |
| **ResNet-34** |                    | :heavy_check_mark: | :heavy_check_mark: |                    |         **142.77**      | **7.00**  |
| **ResNet-50** | :heavy_check_mark: |                    |                    | :heavy_check_mark: |         **295.34**      | **3.38**  |
| **ResNet-50** |                    | :heavy_check_mark: |                    | :heavy_check_mark: |         **144.18**      | **6.93**  |
| **ResNet-101**| :heavy_check_mark: |                    |                    | :heavy_check_mark: |         **531.26**      | **1.88**  |
| **ResNet-101**|                    | :heavy_check_mark: |                    | :heavy_check_mark: |         **422.24**      | **2.36**  |


### Memory Report

- The stats are from the files in `memory_reports/` folder.

- Calculated using the [pytorch-memlab](https://pypi.org/project/pytorch-memlab/) package.


|     Model     |     Cityscapes     |    PASCAL-VOC-2012    |     BasicBlock     |     Bottleneck     |  Allocated Memory (MB)  | # Tensors (M)  |
| ------------- | ------------------ | --------------------- | ------------------ | ------------------ | ----------------------- | -------------- |
| **ResNet-18** | :heavy_check_mark: |                       | :heavy_check_mark: |                    |   **68.69**             |   **17.990**   |
| **ResNet-18** |                    | :heavy_check_mark:    | :heavy_check_mark: |                    |   **46.60**             |   **12.200**   |
| **ResNet-34** | :heavy_check_mark: |                       | :heavy_check_mark: |                    |   **108.16**            |   **28.106**   |
| **ResNet-34** |                    | :heavy_check_mark:    | :heavy_check_mark: |                    |   **85.20**             |   **22.315**   |
| **ResNet-50** | :heavy_check_mark: |                       |                    | :heavy_check_mark: |   **121.73**            |   **31.901**   |
| **ResNet-50** |                    | :heavy_check_mark:    |                    | :heavy_check_mark: |   **100.23**            |   **26.111**   |
| **ResNet-101**| :heavy_check_mark: |                       |                    | :heavy_check_mark: |   **194.65**            |   **50.946**   |
| **ResNet-101**|                    | :heavy_check_mark:    |                    | :heavy_check_mark: |   **172.65**            |   **45.155**   |
