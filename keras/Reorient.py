import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

class Reorient:
    def __init__(self):
        self.resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            GlobalAveragePooling2D(input_shape=(7, 7, 2048)),  # Adapt input shape based on ResNet output
            Dense(128, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def reorient_image(self, image):
        img = cv2.resize(image, (224, 224))  # Resize the image to fit ResNet input
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32') / 255

        features = self.resnet_model.predict(img)
        orientation_probabilities = self.model.predict(features)
        predicted_orientation = np.argmax(orientation_probabilities, axis=1)[0]

        # Rotate the image based on the predicted orientation
        rotated_image = self.rotate_image(image, predicted_orientation * 90)

        return rotated_image, predicted_orientation

    def rotate_image(self, image, angle):
        if angle == 90:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated_image = image  # No rotation needed for 0 degrees
        return rotated_image

