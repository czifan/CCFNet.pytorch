# Multi-Scale Context-Guided Lumbar Spine Disease Identification with Coarse-to-fine Localization and Classification

## Introduction
This repository is the office PyTorch implementation of "Multi-Scale Context-Guided Lumbar Spine Disease Identification with Coarse-to-fine Localization and Classification", ISBI 2022 (Oral). 
The link to the paper is coming soon.
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
| SimpleBaseline | ResNet18-MSC | 6.86 | 33.50 | 91.80 | 92.94 | 88.20 | 74.47 | 75.13 |
| SCN | ResNet18 | 26.57 | 42.73 | 88.56 | 88.77 | 89.26 | 71.18 | 71.18 |
| SCN | ResNet18-MSC | 11.62 | 45.43 | 92.79 | 94.13 | 90.16 | 75.94 | 77.64 | 
| CCF-Net (ours) | ResNet18 | 9.51 | 11.20 | 89.69 | 89.23 | 89.32 | 76.23 | 74.05 | 
| CCF-Net (ours) | ResNet18-MSC | 4.83 | 12.00 | 94.75 | 94.71 | 90.88 | 79.16 | 80.50 |

| Method | Backbone | Params (M) | Flops (G) | L-Disc | L-Vertebra | C-Disc | C-Vertebra | Score |
|---|---|---|---|---|---|---|---|---|
| SimpleBaseline | ResNet50 | 34.00 | 51.64 | 89.33 | 90.07 | 90.21 | 76.06 | 74.59 |
| SimpleBaseline | ResNet50-MSC | 45.33 | 124.80 | 94.53 | 94.54 | 90.56 | 76.06 | 78.77 |
| HRNet | W32 | 28.54 | 40.98 | 96.19 | 95.63 | 89.68 | 78.43 | 80.62 |
| CCF-Net (ours) | ResNet50 | 23.60 | 21.49 | 90.43 | 90.24 | 90.06 | 76.35 | 75.13 |
| CCF-Net (ours) | ResNet50-MSC | 10.79 | 22.09 | 85.68 | 96.41 | 90.59 | 77.46 | 80.64 |

## Citation
Coming soon.
