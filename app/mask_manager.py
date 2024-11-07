# mask_manager.py
import os
import numpy as np
import cv2
import torch
from loguru import logger as guru
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class ClickableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.click_callback = None  # Callback function to be called on click

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.click_callback is not None:
                self.click_callback(event)
        super().mousePressEvent(event)

class MaskManager(QWidget):
    def __init__(self, checkpoint_path, config_path, mask_dir, img_dir):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.image_list = [
            f for f in sorted(os.listdir(self.img_dir))
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.current_index = 0
        self.current_image = None
        self.current_mask = None
        self.sam_model = None
        self.predictor = None
        self.input_points = []
        self.input_labels = []
        self.label_toggle = 1  # Start with positive point (1)
        self.image_name = None  # Name of the current image

        # Initialize UI
        self.init_ui()
        # Initialize SAM2 model and predictor
        self.init_sam_model()
        # Load the first image
        self.load_current_image()

    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()

        # Clickable image label
        self.image_label = ClickableImageLabel()
        # Set the click callback function
        self.image_label.click_callback = self.on_image_clicked
        layout.addWidget(self.image_label)

        # Navigation and Reset Buttons Layout
        button_layout = QHBoxLayout()

        # Previous Image Button
        self.prev_button = QPushButton("< Previous Image")
        self.prev_button.clicked.connect(self.prev_image)
        button_layout.addWidget(self.prev_button, alignment=Qt.AlignLeft)

        # Reset Mask Button
        self.reset_button = QPushButton("Reset Mask")
        self.reset_button.clicked.connect(self.reset_mask)
        button_layout.addWidget(self.reset_button, alignment=Qt.AlignCenter)

        # Next Image Button
        self.next_button = QPushButton("Next Image >")
        self.next_button.clicked.connect(self.next_image)
        button_layout.addWidget(self.next_button, alignment=Qt.AlignRight)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def init_sam_model(self):
        """Initialize the SAM2 model and predictor."""
        if self.sam_model is None:
            self.sam_model = build_sam2(self.config_path, self.checkpoint_path)
            self.predictor = SAM2ImagePredictor(self.sam_model)
            guru.info(f"SAM2 model loaded with checkpoint: {self.checkpoint_path}")

    def load_current_image(self):
        """Load and display the current image."""
        self.input_points = []
        self.input_labels = []
        self.label_toggle = 1  # Reset label toggle
        self.current_mask = None

        if self.image_list and 0 <= self.current_index < len(self.image_list):
            self.image_name = self.image_list[self.current_index]
            image_path = os.path.join(self.img_dir, self.image_name)
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                QMessageBox.warning(self, "Error", f"Failed to load image {self.image_name}")
                return

            # Load the mask if it exists
            mask_path = os.path.join(self.mask_dir, f"mask_{self.image_name}")
            if os.path.exists(mask_path):
                self.current_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                print(f"Loaded existing mask from {mask_path}")
            else:
                # If no mask exists, initialize with a white mask
                self.current_mask = np.ones(self.current_image.shape[:2], dtype=np.uint8) * 255

            # Display the image with the mask overlaid
            self.display_image_with_mask()
        else:
            QMessageBox.warning(self, "Error", "No images to display.")

    def display_image(self, image):
        """Display the given image in the QLabel."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_image_clicked(self, event):
        """Handle image click events with position correction."""
        if self.current_image is not None:
            # Convert global position to position relative to the QLabel
            label_pos = self.image_label.mapFromGlobal(event.globalPos())

            # Get pixmap dimensions
            pixmap = self.image_label.pixmap()
            if pixmap is None:
                return

            pixmap_width = pixmap.width()
            pixmap_height = pixmap.height()

            # QLabel dimensions
            label_width = self.image_label.width()
            label_height = self.image_label.height()

            # Calculate offsets (centered image)
            offset_x = (label_width - pixmap_width) / 2
            offset_y = (label_height - pixmap_height) / 2

            # Check if click is within the displayed image
            if (offset_x <= label_pos.x() <= offset_x + pixmap_width) and \
               (offset_y <= label_pos.y() <= offset_y + pixmap_height):

                # Calculate scaling ratios
                scale_x = self.current_image.shape[1] / pixmap_width
                scale_y = self.current_image.shape[0] / pixmap_height

                # Convert click position to image coordinates
                corrected_x = int((label_pos.x() - offset_x) * scale_x)
                corrected_y = int((label_pos.y() - offset_y) * scale_y)

                # Limit coordinates to image dimensions
                corrected_x = max(0, min(corrected_x, self.current_image.shape[1] - 1))
                corrected_y = max(0, min(corrected_y, self.current_image.shape[0] - 1))

                # Add point
                self.input_points.append([corrected_x, corrected_y])
                self.input_labels.append(self.label_toggle)

                # Toggle label between positive and negative
                self.label_toggle = 1 - self.label_toggle

                # Generate and display the mask
                self.generate_mask()
            else:
                # Click was outside the image area
                pass  # Do nothing or provide feedback if desired

    def process_single_image(self, image, image_name, point_coords, point_labels):
        """
        Generate a mask for a single image and save it to the specified mask directory.
        """
        self.predictor.set_image(image)

        # Generate mask with SAM2
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False
            )

            # Save the mask
            mask_output_path = os.path.join(self.mask_dir, f"mask_{image_name}")
            inverted_mask = 1 - masks[0]
            mask_to_save = (inverted_mask * 255).astype(np.uint8)
            cv2.imwrite(mask_output_path, mask_to_save)
            guru.info(f"Mask saved to {mask_output_path}")

            return inverted_mask * 255

    def generate_mask(self):
        """Generate and display the mask."""
        if self.current_image is None or not self.input_points:
            return

        point_coords = np.array(self.input_points)
        point_labels = np.array(self.input_labels)

        # Generate the mask
        mask = self.process_single_image(
            self.current_image, self.image_name, point_coords, point_labels
        )
        self.current_mask = mask

        # Display the image with the mask overlaid
        self.display_image_with_mask()

    def display_image_with_mask(self):
        """Overlay the mask on the image and display it, blending only on black mask areas."""
        if self.current_image is not None and self.current_mask is not None:
            # Convert mask to binary: black regions (mask==0), white regions (mask==255)
            black_region_mask = cv2.threshold(self.current_mask, 0, 255, cv2.THRESH_BINARY_INV)[1]
            black_region_mask_3ch = cv2.cvtColor(black_region_mask, cv2.COLOR_GRAY2BGR)
            black_region_mask_float = black_region_mask_3ch / 255.0  # Normalize to 0-1

            # Create a blue overlay for the black regions
            blue_overlay = np.zeros_like(self.current_image)
            blue_overlay[:, :, 0] = 255  # Blue channel

            # Blend only on black regions of the mask
            overlay = self.current_image.copy()
            overlay = overlay * (1 - black_region_mask_float) + \
                      (self.current_image * 0.3 + blue_overlay * 0.7) * black_region_mask_float
            overlay = overlay.astype(np.uint8)

            # Display selected points on the overlay
            for idx, point in enumerate(self.input_points):
                color = (0, 255, 0) if self.input_labels[idx] == 1 else (0, 0, 255)
                cv2.circle(overlay, (point[0], point[1]), radius=5, color=color, thickness=-1)

            self.display_image(overlay)
        elif self.current_image is not None:
            # If no mask, just display the image
            self.display_image(self.current_image)

    def reset_mask(self):
        """Reset the current mask to a blank white mask and update the display."""
        if self.current_image is None:
            QMessageBox.warning(self, "Error", "No image loaded to reset the mask.")
            return

        # Reset mask
        self.current_mask = np.ones(self.current_image.shape[:2], dtype=np.uint8) * 255

        # Save the reset mask
        self.save_current_mask()

        # Clear points
        self.clear_points()

        # Display the reset mask
        self.display_image_with_mask()
        QMessageBox.information(self, "Reset Mask", "The mask has been reset to a blank state.")

    def save_current_mask(self):
        """Save the current mask to a file."""
        if self.current_mask is not None:
            mask_output_path = os.path.join(self.mask_dir, f"mask_{self.image_name}")
            mask_to_save = self.current_mask.astype(np.uint8)
            cv2.imwrite(mask_output_path, mask_to_save)
            guru.info(f"Mask saved to {mask_output_path}")

    def clear_points(self):
        """Clear the list of points and labels."""
        self.input_points.clear()
        self.input_labels.clear()
        self.label_toggle = 1  # Reset to positive point

    def next_image(self):
        """Move to the next image."""
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.load_current_image()

    def prev_image(self):
        """Move to the previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
