import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QMessageBox, QDialog, QLineEdit, QDialogButtonBox, QCheckBox, QComboBox
from PyQt5.QtGui import QPixmap, QImage, QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt
from opensfm import dataset
from opensfm.actions import detect_features
from opensfm import features
import yaml

import yaml
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDialogButtonBox, 
    QCheckBox, QComboBox, QLineEdit, QMessageBox, QSizePolicy
)

class ConfigDialog(QDialog):
    # 記憶する位置をクラス変数で保持
    dialog_position = None

    def __init__(self, config_path, feature_type=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Feature Extraction Configuration")
        self.config_path = config_path
        self.feature_type = feature_type  # 初期選択の特徴点タイプ

        # Load config data
        with open(config_path, "r") as f:
            self.config_data = yaml.safe_load(f)

        # Main layout for the dialog
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.fields = {}

        # Set feature type dropdown
        self.set_feature_type_field()

        # Show feature-specific fields initially based on the current feature type
        self.dynamic_layout = QVBoxLayout()  # Separate layout for dynamic fields
        self.main_layout.addLayout(self.dynamic_layout)
        self.update_fields_for_feature_type(self.feature_type or self.config_data["feature_type"])

        # OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.save_config)
        button_box.rejected.connect(self.reject)
        self.main_layout.addWidget(button_box)

        # ウィンドウの位置を設定
        self.set_dialog_position()

    def set_feature_type_field(self):
        """Set dropdown to select feature type and connect change event."""
        label = QLabel("feature_type")
        field = QComboBox()
        feature_types = ["AKAZE", "SURF", "SIFT", "HAHOG", "ORB", "ALIKED"]
        field.addItems(feature_types)
        field.setCurrentText(self.feature_type or self.config_data.get("feature_type", "AKAZE"))
        
        # Connect to update fields when feature type changes
        field.currentTextChanged.connect(self.on_feature_type_change)

        # Add to layout
        row_layout = QHBoxLayout()
        row_layout.addWidget(label)
        row_layout.addWidget(field)
        self.main_layout.addLayout(row_layout)
        self.fields["feature_type"] = field

    def clear_existing_fields(self):
        """Clear any fields already added to avoid duplicates."""
        while self.dynamic_layout.count():
            item = self.dynamic_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        keys_to_remove = [key for key in self.fields if key != "feature_type"]
        for key in keys_to_remove:
            del self.fields[key]

    def on_feature_type_change(self, new_feature_type):
        """Handle feature type change by closing and reopening the dialog with new settings."""
        # 現在の位置を保存
        ConfigDialog.dialog_position = self.pos()
        
        # 新しい特徴点タイプに基づいてダイアログを再生成
        self.close()
        new_dialog = ConfigDialog(self.config_path, feature_type=new_feature_type, parent=self.parent())
        new_dialog.exec_()

    def set_dialog_position(self):
        """記憶した位置にダイアログを開く。位置が未設定ならデフォルト位置で開く。"""
        if ConfigDialog.dialog_position:
            self.move(ConfigDialog.dialog_position)

    def update_fields_for_feature_type(self, feature_type):
        """Update fields displayed based on the selected feature type."""
        self.clear_existing_fields()
        self.add_shared_fields()

        if feature_type == "SIFT":
            self.add_sift_fields()
        elif feature_type == "SURF":
            self.add_surf_fields()
        elif feature_type == "AKAZE":
            self.add_akaze_fields()
        elif feature_type == "HAHOG":
            self.add_hahog_fields()

    def add_shared_fields(self):
        """Add fields that are shared across all feature types."""
        shared_fields = {
            "feature_root": bool,
            "feature_min_frames": int,
            "feature_process_size": int,
            "feature_use_adaptive_suppression": bool,
            "features_bake_segmentation": bool,
        }
        for key, field_type in shared_fields.items():
            label = QLabel(key)
            if field_type == bool:
                field = QCheckBox()
                field.setChecked(self.config_data.get(key, False))
            elif field_type == int:
                field = QLineEdit(str(self.config_data.get(key, 0)))
                field.setValidator(QIntValidator())
            else:
                field = QLineEdit(str(self.config_data.get(key, "")))

            row_layout = QHBoxLayout()
            row_layout.addWidget(label)
            row_layout.addWidget(field)
            self.dynamic_layout.addLayout(row_layout)
            self.fields[key] = field

    def add_sift_fields(self):
        self.add_field("sift_peak_threshold", float, 0.1)
        self.add_field("sift_edge_threshold", int, 10)

    def add_surf_fields(self):
        self.add_field("surf_hessian_threshold", int, 3000)
        self.add_field("surf_n_octaves", int, 4)
        self.add_field("surf_n_octavelayers", int, 2)
        self.add_field("surf_upright", bool)

    def add_akaze_fields(self):
        self.add_field("akaze_omax", int, 4)
        self.add_field("akaze_dthreshold", float, 0.001)
        self.add_field("akaze_descriptor", str, "MSURF")
        self.add_field("akaze_descriptor_size", int, 0)
        self.add_field("akaze_descriptor_channels", int, 3)
        self.add_field("akaze_kcontrast_percentile", float, 0.7)
        self.add_field("akaze_use_isotropic_diffusion", bool)

    def add_hahog_fields(self):
        self.add_field("hahog_peak_threshold", float, 0.00001)
        self.add_field("hahog_edge_threshold", int, 10)
        self.add_field("hahog_normalize_to_uchar", bool)

    def add_field(self, key, field_type, default_value=None):
        label = QLabel(key)
        if field_type == bool:
            field = QCheckBox()
            field.setChecked(self.config_data.get(key, default_value))
        elif field_type == int:
            field = QLineEdit(str(self.config_data.get(key, default_value)))
            field.setValidator(QIntValidator())
        elif field_type == float:
            field = QLineEdit(str(self.config_data.get(key, default_value)))
            field.setValidator(QDoubleValidator())
        else:
            field = QLineEdit(str(self.config_data.get(key, default_value)))

        row_layout = QHBoxLayout()
        row_layout.addWidget(label)
        row_layout.addWidget(field)
        self.dynamic_layout.addLayout(row_layout)
        self.fields[key] = field

    def save_config(self):
        for key, field in self.fields.items():
            if isinstance(field, QCheckBox):
                self.config_data[key] = field.isChecked()
            elif isinstance(field, QLineEdit):
                text = field.text()
                self.config_data[key] = int(text) if text.isdigit() else float(text) if '.' in text else text
            elif isinstance(field, QComboBox):
                self.config_data[key] = field.currentText()

        with open(self.config_path, "w") as f:
            yaml.safe_dump(self.config_data, f)
        QMessageBox.information(self, "Config", "Configuration saved successfully.")
        self.accept()

class FeatureExtractor(QWidget):
    def __init__(self, workdir, image_list):
        super().__init__()
        self.workdir = workdir
        self.image_list = image_list
        self.dataset = dataset.DataSet(workdir)
        self.current_image = None  # 現在の画像データを保持する変数
        self.config_path = os.path.join(workdir, "config.yaml")
        self.config_data = self.load_config_data(self.config_path)

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
        config_path = os.path.join(self.workdir, "config.yaml")
        dialog = ConfigDialog(config_path=config_path, feature_type=self.config_data.get("feature_type", "AKAZE"), parent=self)
        if dialog.exec_() == QDialog.Accepted:
            QMessageBox.information(self, "Config", "Configuration saved successfully.")
            self.dataset = dataset.DataSet(self.workdir)

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

    def load_config_data(self, config_path):
        """Load configuration data from a YAML file."""
        import yaml
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return {}