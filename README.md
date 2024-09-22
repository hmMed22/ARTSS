# ARTSS (Automated Radiographic Tool for Sharp Score prediction)

ARTSS is a Python package designed for automatically predicting sharp scores on hand X-ray images. It provides functionality for reorienting images, hand segmentation, joint identification, and total sharp score prediction using deep learning models.

## Features

- **Reorientation**: Utilizes ResNet for rotating X-ray images to a 90-degree orientation.
- **Hand Segmentation**: Uses U-Net for segmenting hands from X-ray images.
- **Joint Identification**: Implements YOLOv7 for identifying joints in the segmented hand images.
- **Sharp Score Prediction**: Uses VGG16, VGG19, ...,and ViT for predicting total sharp scores based on identified joints.

## Installation

You can install the ARTSS package using pip:

```bash
pip install ARTSS


=======
# ARTSS
A reliable automated radiographic total sharp scoring (ARTSS) framework using Four-stage deep learning for full-hand X-ray images.
