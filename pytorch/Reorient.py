import cv2
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF

class Reorient(nn.Module):
    def __init__(self):
        super(Reorient, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Removing the last fully connected layer from the ResNet model
        self.resnet_model = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.resnet_model(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def angle_image(self, image):
        img = cv2.resize(image, (224, 224))  # Resize the image to fit ResNet input
        img = np.transpose(img, (2, 0, 1))   # PyTorch uses channel-first format
        img = torch.from_numpy(img).float() / 255.0

        # Add batch dimension
        img = img.unsqueeze(0)

        with torch.no_grad():
            features = self.forward(img)
            orientation_probabilities = F.softmax(features, dim=1)
            predicted_orientation = torch.argmax(orientation_probabilities, dim=1).item()

        return predicted_orientation

    def rotate_image(self, image):
        angle = self.angle_image(image)
        rotation_required = 90 - angle  # Calculate the rotation required from the predicted angle

        # Rotate the image to achieve the desired orientation (90 degrees, fingers upward)
        if rotation_required == 90:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_required == 180:
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        elif rotation_required == 270:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated_image = image  # No rotation needed for 0 degrees or multiples of 360

        return rotated_image
