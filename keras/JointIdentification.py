import argparse
import time
import os
from pathlib import Path
import cv2
import torch
import logging
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, increment_path, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class YOLOv7Detector:
    def __init__(self, weights, img_size=640, conf_thres=0.25, iou_thres=0.45, device='cpu', save_dir='runs/detect'):
        """
        Initialize YOLOv7 Detector.

        Args:
            weights (str): Path to the YOLOv7 model weights.
            img_size (int): Input image size for inference.
            conf_thres (float): Confidence threshold for detections.
            iou_thres (float): IOU threshold for NMS.
            device (str): Device for inference ('cpu' or 'cuda').
            save_dir (str): Directory to save detection results.
        """
        self.device = select_device(device)
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.save_dir = Path(increment_path(Path(save_dir) / 'exp'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Load YOLOv7 model
        self.model = attempt_load(weights, map_location=self.device)
        self.model = TracedModel(self.model, self.device, img_size)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def detect_and_crop(self, source, save_crops=True):
        """
        Perform object detection and save cropped regions.

        Args:
            source (str): Source of images/videos for detection.
            save_crops (bool): Save cropped detections.
        """
        dataset = LoadImages(source, img_size=self.img_size, stride=int(self.model.stride.max()))
        logger.info(f"Running inference on: {source}")
        crop_dir = self.save_dir / 'crops'
        crop_dir.mkdir(parents=True, exist_ok=True)

        # Process dataset
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.float() / 255.0  # Normalize
            img = img.unsqueeze(0) if img.ndimension() == 3 else img

            # Inference
            pred = self.model(img, augment=False)[0]
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

            # Process detections
            for det in pred:
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                    self.save_results(im0s, det, crop_dir if save_crops else None)

    def save_results(self, img, det, crop_dir=None):
        """
        Save detection results and optionally cropped objects.

        Args:
            img (numpy.ndarray): Original image.
            det (torch.Tensor): Detection results.
            crop_dir (Path): Directory to save cropped regions.
        """
        for *xyxy, conf, cls in reversed(det):
            label = f"{self.names[int(cls)]} {conf:.2f}"
            plot_one_box(xyxy, img, label=label, color=self.colors[int(cls)], line_thickness=2)

            if crop_dir:
                crop_path = crop_dir / f"{int(cls)}_{time.time()}.jpg"
                x_min, y_min, x_max, y_max = map(int, xyxy)
                cropped_img = img[y_min:y_max, x_min:x_max]
                cv2.imwrite(str(crop_path), cropped_img)
                logger.info(f"Cropped region saved: {crop_path}")


def main(opt):
    """
    Main function for YOLOv7 inference and cropping.

    Args:
        opt (argparse.Namespace): Parsed command-line arguments.
    """
    detector = YOLOv7Detector(
        weights=opt.weights,
        img_size=opt.img_size,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        device=opt.device,
        save_dir=opt.project,
    )
    detector.detect_and_crop(source=opt.source, save_crops=not opt.nosave)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/images', help='Source file/folder')
    parser.add_argument('--img-size', type=int, default=640, help='Inference image size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='Inference device ("cpu" or "cuda")')
    parser.add_argument('--nosave', action='store_true', help='Do not save cropped regions')
    parser.add_argument('--project', default='runs/detect', help='Save directory')
    opt = parser.parse_args()

    with torch.no_grad():
        main(opt)
