'''
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
    
'''
import cv2
import numpy as np
from pathlib import Path

class ImageRotator:
    def __init__(self, reference_image_path, output_path, angle_tolerance=10):
        """
        Initializes the ImageRotator class.

        :param reference_image_path: Path to the reference image.
        :param output_path: Directory to save the rotated images.
        :param angle_tolerance: Tolerance for angle alignment in degrees.
        """
        self.reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
        self.output_path = output_path
        self.angle_tolerance = angle_tolerance

    @staticmethod
    def detect_orientation(image):
        """
        Detects the orientation of the image using edge detection and contour moments.

        :param image: Input grayscale image.
        :return: Rotation angle to align the image to its vertical orientation.
        """
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0  # Return 0 if no contours are found

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Fit a rotated rectangle around the contour
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[-1]

        # Correct the angle to align the longer edge vertically
        if angle < -45:
            angle = 90 + angle

        return angle

    @staticmethod
    def rotate_image(image, angle):
        """
        Rotates the image to the specified angle.

        :param image: Input image.
        :param angle: Angle in degrees to rotate the image.
        :return: Rotated image.
        """
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Compute the rotation matrix and rotate the image
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
        return rotated

    def process_images(self, input_dir):
        """
        Processes all images in the input directory, aligning them to the reference format.

        :param input_dir: Directory containing input images.
        """
        input_dir = Path(input_dir)
        output_dir = Path(self.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        for image_path in input_dir.glob("*.png"):  # Change extension as needed
            print(f"Processing {image_path.name}...")
            
            # Read the image
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            # Detect orientation and rotate
            angle = self.detect_orientation(image)
            if abs(angle) > self.angle_tolerance:
                rotated_image = self.rotate_image(image, angle)
            else:
                rotated_image = image  # No rotation needed

            # Save the rotated image
            output_path = output_dir / image_path.name
            cv2.imwrite(str(output_path), rotated_image)
            print(f"Saved aligned image to {output_path}")

# Example usage:
if __name__ == "__main__":
    reference_image_path =   # Path to the reference image
    input_dir = #  Path to the input_images
    output_dir = # Path to the output_images

    rotator = ImageRotator(reference_image_path=reference_image_path, output_path=output_dir)
    rotator.process_images(input_dir)


