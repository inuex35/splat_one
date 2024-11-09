import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from opensfm import dataset
from opensfm.actions import detect_features
from opensfm import features

class FeatureExtractor(QWidget):
    def __init__(self, workdir, image_list):
        super().__init__()
        self.workdir = workdir
        self.image_list = image_list
        self.dataset = dataset.DataSet(workdir)
        self.current_image = None  # 現在の画像データを保持する変数

        # UI setup with a vertical layout
        layout = QVBoxLayout()

        # 上部に空間を追加して中央に配置
        layout.addStretch(1)

        # Display label for feature visualization, centered vertically
        self.display_label = QLabel("Select an image to view features.")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.display_label, alignment=Qt.AlignVCenter)

        # 下部に空間を追加して中央に配置
        layout.addStretch(1)

        button_layout = QHBoxLayout()

        # Previous Image Button
        self.extract_button = QPushButton("Extract Features")
        self.extract_button.clicked.connect(self.run_detect_features)
        button_layout.addWidget(self.extract_button)

        # Reset Mask Button
        self.config_button = QPushButton("Config")
        self.config_button.clicked.connect(self.configure_features)
        button_layout.addWidget(self.config_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def resizeEvent(self, event):
        """Handle the resize event to adjust the image size to fit the QLabel area."""
        if self.current_image is not None:
            self.set_image_to_label(self.current_image)  # 現在の画像をリサイズして再表示

    def load_image_by_name(self, image_name):
        """Load an image by name, and display features if available, or show the original image."""
        try:
            features_data = self.dataset.load_features(image_name)
            self.plot_features(image_name, features_data)
        except FileNotFoundError:
            self.show_original_image(image_name)

    def run_detect_features(self):
        """Run feature detection on all images in the dataset."""
        detect_features.run_dataset(self.dataset)
        QMessageBox.information(self, "Feature Extraction", "Feature extraction completed for all images.")

    def configure_features(self):
        """Open the configuration dialog for feature extraction."""
        QMessageBox.information(self, "Feature Extraction", "Feature extraction configuration dialog.")

    def show_original_image(self, image_name):
        """Show the original image in QLabel when no features are available."""
        image_path = os.path.join(self.workdir, "images", image_name)
        image = cv2.imread(image_path)
        if image is not None:
            self.current_image = image  # 現在の画像を保持
            self.set_image_to_label(image)
        else:
            self.display_label.setText("Image not found.")

    def plot_features(self, image_name, features_data):
        """Use OpenCV to plot features of the selected image and display in QLabel."""
        # Load image
        image = self.dataset.load_image(image_name)
        h, w, _ = image.shape
        pixels = features.denormalized_image_coordinates(features_data.points, w, h)
        fixed_size = 5  # 円の固定半径

        for p in pixels:
            center = (int(p[0]), int(p[1]))
            cv2.circle(image, center, fixed_size, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)  # Yellow circles

        # 現在の画像を保持し、ラベルに設定
        self.current_image = image  
        self.set_image_to_label(image)

    def set_image_to_label(self, rgb_image):
        """Resize the image to fit the QLabel width while maintaining aspect ratio and display it."""
        label_width = self.display_label.width()
        h, w, _ = rgb_image.shape
        aspect_ratio = h / w  # アスペクト比を計算

        # 横幅を QLabel の幅に合わせ、アスペクト比を維持した高さを計算
        new_width = label_width
        new_height = int(new_width * aspect_ratio)

        # 画像をリサイズ
        resized_image = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to QPixmap and set to QLabel
        height, width, channel = resized_image.shape
        bytes_per_line = channel * width
        q_image = QImage(resized_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.display_label.setPixmap(pixmap)