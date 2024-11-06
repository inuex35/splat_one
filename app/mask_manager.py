# mask_manager.py
import os
import numpy as np
import cv2
from sam2mask.prompt_gui import PromptGUI
from PyQt5.QtWidgets import QMessageBox

class MaskManager:
    def __init__(self, checkpoint_path, config_path, mask_dir):
        self.prompt_gui = PromptGUI(checkpoint_path, config_path)
        self.mask_dir = mask_dir
        self.current_mask = None
        self.input_points = []
        self.input_labels = []
        self.label_toggle = 1

    def reset_mask(self, image_shape):
        """白いマスクで初期化"""
        self.current_mask = np.ones(image_shape, dtype=np.uint8) * 255
        return self.current_mask

    def generate_mask(self, image, image_name):
        """クリックされた点の座標とラベルでマスクを生成"""
        new_mask = self.prompt_gui.process_single_image(
            image,
            self.mask_dir,
            image_name,
            point_coords=np.array(self.input_points),
            point_labels=np.array(self.input_labels),
        )

        if new_mask is None:
            return None  # エラー処理は呼び出し元で行う

        if self.current_mask is None:
            self.current_mask = new_mask
        else:
            if new_mask.shape != self.current_mask.shape:
                new_mask = cv2.resize(new_mask, (self.current_mask.shape[1], self.current_mask.shape[0]))
            if new_mask.dtype != self.current_mask.dtype:
                new_mask = new_mask.astype(self.current_mask.dtype)
            self.current_mask = cv2.bitwise_and(self.current_mask, new_mask)

        return self.current_mask

    def save_current_mask(self, image_name):
        """マスクを保存"""
        mask_output_path = os.path.join(self.mask_dir, f"mask_{image_name}")
        cv2.imwrite(mask_output_path, self.current_mask)
        return mask_output_path

    def add_point(self, point):
        """マスク生成のためのポイントを追加"""
        self.input_points.append(point)
        self.input_labels.append(self.label_toggle)
        self.label_toggle = 1 - self.label_toggle  # Toggle label

    def clear_points(self):
        """座標リストとラベルをクリア"""
        self.input_points.clear()
        self.input_labels.clear()
