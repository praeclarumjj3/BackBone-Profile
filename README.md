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

This repo contains the folowing **Performance Stats** for a few popularly used **backbone networks** in the field of Computer Vision:

```
- Inference Time on a GTX 2080Ti

- Inference Time on a TitanXP

- Infernce Time on a CPU

- Memory Report during Inference

- Model Structures
```

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

The repository contains the following architecture:

- [MobileNetV2 Profiler](https://github.com/praeclarumjj3/BackBone-Profile/tree/master/MobilnetV2%20Profiler) - Scripts and stats for inference performance of **MobileNet-V2**.

- [ResNet Profiler](https://github.com/praeclarumjj3/BackBone-Profile/tree/master/ResNet%20Profiler) - Scripts and stats for inference performance of various variants of **ResNet**.

- [Xception Profiler](https://github.com/praeclarumjj3/BackBone-Profile/tree/master/Xception%20Profiler) - Scripts and stats for inference performance of **Xception**.

## 4. Reproduction

- Refer to the **README.md** of the corresponding architectures.

## 5. Results

All the experiments are performed with a `batch size=1` and 300 iterations.

#### Performance on Cityscapes

|     Model         | Inference Time (ms) [2080Ti] | Inference Time (ms) [TitanXP]|   FPS [2080Ti] | FPS [TitanXP]  |  Allocated Memory (MB)  | # Params (M)  |
| ----------------- | ---------------------------- | ---------------------------- | -------------- | -------------- | ----------------------- | --------------|
| **ResNet-18**     |       **18.719**             |       **23.622**             | **53.42**      | **42.33**      |   **68.69**             | **11.689**    |
| **ResNet-34**     |       **31.779**             |       **38.588**             | **31.46**      | **25.91**      |   **108.16**            | **21.797**    |
| **ResNet-50**     |       **61.397**             |       **82.334**             | **16.28**      | **12.14**      |   **121.73**            | **25.557**    |
| **ResNet-101**    |       **100.426**            |       **122.491**            | **9.95**       | **8.16**       |   **194.65**            | **44.549**    |
| **MobileNet-V2**  |       **33.627**             |       **54.314**             | **29.73**      | **18.41**      |   **37.58**             | **3.504**     |
| **Xception**      |       **77.079**             |       **144.919**            | **12.97**      | **6.90**       |   **111.45**            | **22.855**     |

#### Performance on PASCAL-VOC-2012

|     Model         | Inference Time (ms) [2080Ti] | Inference Time (ms) [TitanXP]|   FPS [2080Ti] | FPS [TitanXP]  |  Allocated Memory (MB)  | # Params (M)  | 
| ----------------- | ---------------------------- | ---------------------------- | -------------- | -------------- | ----------------------- | --------------|
| **ResNet-18**     |       **2.547**              |       **2.940**              | **392.61**     | **340.13**     |   **46.60**             | **11.689**    |
| **ResNet-34**     |       **5.197**              |       **4.959**              | **192.41**     | **201.65**     |   **85.20**             | **21.797**    |
| **ResNet-50**     |       **7.628**              |       **8.927**              | **131.09**     | **112.01**     |   **100.23**            | **25.557**    |
| **ResNet-101**    |       **12.579**             |       **14.509**             | **79.49**      | **68.92**      |   **172.65**            | **44.549**    |
| **MobileNet-V2**  |       **5.570**              |       **5.795**              | **179.53**     | **172.56**     |   **14.49**             | **3.504**     |
| **Xception**      |       **7.919**              |       **12.042**             | **126.27**     | **83.04**      |   **89.36**             | **22.855**     |

### Inference Time (CPU) 

#### Performance on Cityscapes

|     Model         | Inference Time (ms) |    FPS    | 
| ----------------- | --------------------| --------- |
| **ResNet-18**     |       **566.75**    | **1.76**  |
| **ResNet-34**     |       **807.57**    | **1.23**  |
| **ResNet-50**     |       **1626.05**   | **0.61**  |
| **ResNet-101**    |       **2344.98**   | **0.42**  |
| **MobileNet-V2**  |       **560.022**   | **1.78**  |
| **Xception**      |       **2782.874**  | **0.35**  |

#### Performance on PASCAL-VOC-2012

|        Model          |    Inference Time (ms)    |       FPS        | 
|    -----------------  |    --------------------   |    ---------     |
|    **ResNet-18**      |          **55.79**        |    **17.92**     |
|    **ResNet-34**      |          **78.77**        |    **12.69**     |
|    **ResNet-50**      |          **133.71**       |    **7.47**      |
|    **ResNet-101**     |          **223.59**       |    **4.47**      |
|    **MobileNet-V2**   |          **70.180**       |   **14.24**      | 
|    **Xception**       |          **229.000**      |   **4.36**       |