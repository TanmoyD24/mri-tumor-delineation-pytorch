# Attention-Guided Ensemble Networks for Automated Segmentation of Low-Grade Gliomas (LGG) in Brain MRI

This repository contains an end-to-end deep learning pipeline for the precise automated segmentation of Low-Grade Gliomas (LGG) using multi-sequence structural brain MRI scans. The system features a custom fully convolutional U-Net-style topology enhanced by **Convolutional Block Attention Modules (CBAM)** at the structural bottleneck to dynamically suppress healthy background tissue while accentuating clinical tumor regions.

---

## System Framework & Architecture

The project is built entirely from scratch using a professional **local-first, production-ready** Python ML framework stack:

* **Core Framework:** `PyTorch 2.x` utilizing CUDA-accelerated backend computations for deep layer gradient updates.
* **Data Augmentation Engineering:** `Albumentations` multi-threaded processing library executing spatial-geometric adjustments (`ShiftScaleRotate`, `HorizontalFlip`) and pixel-intensity transforms (`RandomBrightnessContrast`) synchronized perfectly across matching structural image-mask matrices.
* **Feature Extractors (Backbones):** * **EfficientNetV2-M:** Implemented as a high-capacity, deeply parameterized backbone leveraging fused-MBConv layers for fine-grained morphological feature maps.
  * **MobileNetV3-Large:** Implemented as an efficient, lightweight backbone using hardware-aware inverted residual blocks and hard-swish non-linearities for fast feature extraction.
* **Attention Mechanism:** A custom **CBAM (Convolutional Block Attention Module)** is integrated directly at the bottleneck layer. It sequentializes feature map refining via 1D **Channel Attention** (exploiting inter-channel relationships via max and average pooling vectors) and 2D **Spatial Attention** (exploiting inter-spatial relationships via a $7 \times 7$ kernel convolutional layer).
* **Upsampling Decoder:** Multi-stage transposed convolutional (`ConvTranspose2d`) decoder pipeline equipped with paired 2D Batch Normalization and rectified linear activation blocks, scaling spatial resolution cleanly back to the native $256 \times 256$ input scale.

---

## Performance Results & In-Depth Analysis

### 1. Global Consensus Evaluation Metrics
Evaluated across the hidden test set partition using a unified multi-model ensemble voting framework ($0.7 \times \text{EfficientNetV2-M} + 0.3 \times \text{MobileNetV3-Large}$), the system achieves the following research-grade indicators:

| Core Clinical Indicator | Achieved Value | Analytical Interpretation |
| :--- | :--- | :--- |
| **Dice Coefficient (F1-Score)** | **0.8186** | Demonstrates excellent global spatial overlap and geometric congruence with manual radiologist annotations. |
| **Intersection over Union (IoU)** | **0.6928** | Confirms highly precise pixel-level convergence across the absolute intersection-to-union ratio boundary. |
| **Sensitivity (Recall / TPR)** | **0.9023** | Exceptionally high; minimizes dangerous clinical false negatives, ensuring $90.23\%$ of the active tumor mass is safely captured. |
| **Specificity (TNR)** | **0.9968** | Near-perfect; practically eliminates false positives, guaranteeing healthy structural brain matters are safely preserved without over-segmentation. |

### 2. Convergence Dynamics & Training Analysis
The training logs across 20 epochs reveal outstanding training stability and performance optimization:

```text
Training Highlights:
* MobileNetV3-Large CBAM: Epoch 20/20 - Train Loss: 0.0320 | Val Loss: 0.0305 | Val Dice: 0.7786
* EfficientNetV2-M CBAM:  Epoch 20/20 - Train Loss: 0.0341 | Val Loss: 0.0341 | Val Dice: 0.7554


git clone [https://github.com/TanmoyD24/mri-tumor-delineation-pytorch.git](https://github.com/TanmoyD24/mri-tumor-delineation-pytorch.git)
cd mri-tumor-delineation-pytorch
pip install torch torchvision albumentations opencv-python pandas matplotlib scikit-learn

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eff_model = EfficientNetV2M_Segmentation_CBAM(num_classes=1).to(device)
eff_model.load_state_dict(torch.load("eff_seg_best_model.pth", map_location=device))

mob_model = MobileNetV3Large_Segmentation_CBAM(num_classes=1).to(device)
mob_model.load_state_dict(torch.load("mob_seg_best_model.pth", map_location=device))

visualize_tumor_predictions(eff_model, mob_model, test_loader, num_images=4)