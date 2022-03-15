# Multi-Scale Context-Guided Lumbar Spine Disease Identification with Coarse-to-fine Localization and Classification

## Introduction
This repository is the office PyTorch implementation of "Multi-Scale Context-Guided Lumbar Spine Disease Identification with Coarse-to-fine Localization and Classification", also called CCF-Net. 
The paper is avaiable in (coming soon).
![image](https://user-images.githubusercontent.com/24490441/158391952-a2841e9a-c8d0-426b-959f-03a92c62e955.png)

## Requirements
- Pytorch>=1.1.0
- CPU or GPU
- Other packages can be installed with the following instruction:
```
pip install requirements.txt
```
## Quick start
Running the code with the following command.
```
python main_ours_resnet50_msc.py
```

## Results
| Method | Backbone | Params (M) | Flops (G) | L-Disc | L-Vertebra | C-Disc | C-Vertebra | Score |
|---|---|---|---|---|---|---|---|---|
| SimpleBaseline | ResNet18 | 15.38 | 33.23 | 87.81 | 86.11 | 89.26 | 71.71 | 70.70 |

