# ARTSS (Automated Radiographic Tool for Sharp Score prediction)

The ARTSS Python package is an ongoing project aimed at automating the prediction of sharp scores on hand X-ray images. It provides functionality for reorienting images, hand segmentation, joint identification, and total sharp score prediction using deep learning models.

## Features

- **Reorientation**: Utilizes ResNet for rotating X-ray images to a 90-degree orientation.
- **Hand Segmentation**: Uses U-Net to segment hands from X-ray images.
- **Joint Identification**: Implements YOLOv7 for identifying joints in the segmented hand images.
- **Sharp Score Prediction**: Uses VGG16, VGG19, ..., and ViT to predict total sharp scores based on identified joints.

## Installation

You can install the ARTSS package using pip:

```bash
pip install ARTSS


=======
# ARTSS
An automated radiographic total sharp scoring (ARTSS) framework using Four-stage deep learning for full-hand X-ray images.
