import cv2
import numpy as np
from tensorflow.keras.models import load_model

class SegmentHand:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def segment_hand(self, image):
        img = cv2.resize(image, (256, 256))
        img = np.expand_dims(img, axis=0)

        # Perform hand segmentation using the loaded model
        predicted_mask = self.model.predict(img)[0]
        predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255

        return predicted_mask
