# sam2mask/prompt_gui.py
import os
import cv2
import numpy as np
import torch
from loguru import logger as guru
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class PromptGUI:
    def __init__(self, checkpoint_path, config_path):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.sam_model = None
        self.predictor = None
        self.init_sam_model()

    def init_sam_model(self):
        """SAM2モデルと予測器の初期化"""
        if self.sam_model is None:
            self.sam_model = build_sam2(self.config_path, self.checkpoint_path)
            self.predictor = SAM2ImagePredictor(self.sam_model)
            guru.info(f"SAM2 model loaded with checkpoint: {self.checkpoint_path}")

    def process_single_image(self, image, mask_dir, image_name, point_coords, point_labels):
        """
        1つの画像に対してマスクを生成し、指定されたマスクディレクトリに保存します。
        """
        self.predictor.set_image(image)

        # SAM2でマスク生成
        print(point_coords)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False
            )

            # マスクを保存
            mask_output_path = os.path.join(mask_dir, f"mask_{image_name}")
            guru.info(f"Mask saved to {mask_output_path}")
            inverted_mask = 1 - masks[0]
            return inverted_mask * 255
