import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QMessageBox,
    QDialog,
    QLineEdit,
    QDialogButtonBox,
    QCheckBox,
    QComboBox,
    QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage, QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from opensfm import dataset
from opensfm.actions import detect_features
from opensfm import features
import yaml
import time

class ConfigDialog(QDialog):
    # Store the remembered position as a class variable
    dialog_position = None

    def __init__(self, config_path, feature_type=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Feature Extraction Configuration")
        self.config_path = config_path
        self.feature_type = feature_type  # Initial selected feature type

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

        # Set the window position
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
        # Save the current position
        ConfigDialog.dialog_position = self.pos()
        
        # Recreate the dialog based on the new feature type
        self.close()
        new_dialog = ConfigDialog(self.config_path, feature_type=new_feature_type, parent=self.parent())
        new_dialog.exec_()

    def set_dialog_position(self):
        """
        Opens the dialog at the remembered position. If the position is not set, opens at the default position.
        """
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
                text = field.text().strip()
                # Convert to number if possible
                if text.replace('.', '', 1).isdigit():
                    # Convert to float if it contains a decimal point, otherwise to int
                    if '.' in text:
                        self.config_data[key] = float(text)
                    else:
                        self.config_data[key] = int(text)
                else:
                    # Treat as string if not a number
                    self.config_data[key] = text
            elif isinstance(field, QComboBox):
                self.config_data[key] = field.currentText()

        with open(self.config_path, "w") as f:
            yaml.safe_dump(self.config_data, f)
        QMessageBox.information(self, "Config", "Configuration saved successfully.")
        self.accept()

class ProgressWindow(QWidget):
    def __init__(self, total_images):
        super().__init__()
        self.setWindowTitle("Processing Progress")
        self.setFixedSize(400, 100)

        layout = QVBoxLayout()
        self.label = QLabel("Processing images...")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(total_images)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def update_progress(self, processed_images):
        self.progress_bar.setValue(processed_images)
        self.label.setText(f"Feature extraction: {processed_images} / {self.progress_bar.maximum()} images")

class ProgressMonitorThread(QThread):
    progress = pyqtSignal(int)
    stopped = pyqtSignal()

    def __init__(self, feature_folder, total_images):
        super().__init__()
        self.feature_folder = feature_folder
        self.total_images = total_images
        self._is_running = True

    def run(self):
        while self._is_running:
            processed_count = self.count_feature_files()
            self.progress.emit(processed_count)
            if processed_count >= self.total_images:
                break
            time.sleep(0.5)  # 0.5秒間隔で更新

        if not self._is_running:
            self.stopped.emit()

    def stop(self):
        self._is_running = False

    def count_feature_files(self):
        return len([f for f in os.listdir(self.feature_folder) if f.endswith('.features.npz')])


class FeatureExtractionThread(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def run(self):
        try:
            detect_features.run_dataset(self.dataset)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class FeatureExtractor(QWidget):
    def __init__(self, workdir, image_list):
        super().__init__()
        self.workdir = workdir
        self.image_list = image_list
        self.dataset = dataset.DataSet(workdir)
        self.current_image = None
        self.config_path = os.path.join(workdir, "config.yaml")
        self.config_data = self.load_config_data(self.config_path)

        layout = QVBoxLayout()
        layout.addStretch(1)

        self.display_label = QLabel("Select an image to view features.")
        self.display_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.display_label)

        layout.addStretch(1)

        self.extract_button = QPushButton("Extract Features")
        self.extract_button.clicked.connect(self.toggle_feature_extraction)
        layout.addWidget(self.extract_button)

        self.setLayout(layout)

        self.progress_window = None
        self.feature_thread = None
        self.monitor_thread = None
        self.processing = False
        self.feature_folder = os.path.join(self.workdir, "features")

    def toggle_feature_extraction(self):
        if not self.processing:
            self.start_feature_extraction()
        else:
            self.stop_feature_extraction()

    def start_feature_extraction(self):
        total_images = len(self.image_list)
        self.progress_window = ProgressWindow(total_images)
        self.progress_window.show()

        # 特徴抽出スレッド開始
        self.feature_thread = FeatureExtractionThread(self.dataset)
        self.feature_thread.finished.connect(self.on_feature_extraction_finished)
        self.feature_thread.error.connect(self.on_feature_extraction_error)
        self.feature_thread.start()

        # 進捗監視スレッド開始
        self.monitor_thread = ProgressMonitorThread(self.feature_folder, total_images)
        self.monitor_thread.progress.connect(self.update_progress_window)
        self.monitor_thread.stopped.connect(self.on_feature_extraction_stopped)
        self.monitor_thread.start()

        self.processing = True
        self.extract_button.setText("Stop Feature Extraction")

    def stop_feature_extraction(self):
        if self.monitor_thread:
            self.monitor_thread.stop()
        self.extract_button.setEnabled(False)
        if self.progress_window:
            self.progress_window.close() 

    def update_progress_window(self, processed_images):
        if self.progress_window:
            self.progress_window.update_progress(processed_images)

    def on_feature_extraction_finished(self):
        if self.monitor_thread:
            self.monitor_thread._is_running = False
        self.processing = False
        self.extract_button.setText("Extract Features")
        self.extract_button.setEnabled(True)
        if self.progress_window:
            self.progress_window.close()  # Progressウィンドウを閉じる
            QMessageBox.information(self, "Completed", "Feature extraction completed.")

    def on_feature_extraction_error(self, error_message):
        if self.monitor_thread:
            self.monitor_thread._is_running = False
        self.processing = False
        self.extract_button.setText("Extract Features")
        self.extract_button.setEnabled(True)
        if self.progress_window:
            self.progress_window.close()  # Progressウィンドウを閉じる
            QMessageBox.critical(self, "Error", f"Error:\n{error_message}")

    def on_feature_extraction_stopped(self):
        self.processing = False
        self.extract_button.setText("Extract Features")
        self.extract_button.setEnabled(True)
        if self.progress_window:
            self.progress_window.close()


    def resizeEvent(self, event):
        """Handle the resize event to adjust the image size to fit the QLabel area."""
        if self.current_image is not None:
            self.set_image_to_label(self.current_image)  # Resize and redisplay the current image

    def load_image_by_name(self, image_name):
        """
        Called when the user selects an image.
        1) Display features if available
        2) Otherwise, display the original image
        """
        # Display "Image not found." if the image does not exist
        image_path = os.path.join(self.workdir, "images", image_name)
        if not os.path.exists(image_path):
            self.display_label.setText("Image not found.")
            return

        try:
            features_data = self.dataset.load_features(image_name)
            self.plot_features(image_name, features_data)
        except FileNotFoundError:
            self.show_original_image(image_name)

    def configure_features(self):
        """Open the configuration dialog for feature extraction."""
        config_path = os.path.join(self.workdir, "config.yaml")
        dialog = ConfigDialog(
            config_path=config_path,
            feature_type=self.config_data.get("feature_type", "AKAZE"),
            parent=self
        )
        if dialog.exec_() == QDialog.Accepted:
            QMessageBox.information(self, "Config", "Configuration saved successfully.")
            # Recreate the dataset as the config has been updated
            self.dataset = dataset.DataSet(self.workdir)

    def show_original_image(self, image_name):
        """Show the original image in QLabel when no features are available."""
        image_path = os.path.join(self.workdir, "images", image_name)
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            self.current_image = image  # Hold the current image
            self.set_image_to_label(image)
        else:
            self.display_label.setText("Image not found.")

    def plot_features(self, image_name, features_data):
        """Use OpenCV to plot features of the selected image and display in QLabel."""
        # Load image
        image = self.dataset.load_image(image_name)
        if image is None:
            self.display_label.setText("Image not found.")
            return

        # Draw features
        h, w, _ = image.shape
        pixels = features.denormalized_image_coordinates(features_data.points, w, h)
        fixed_size = 5  # Fixed radius for circles

        for p in pixels:
            center = (int(p[0]), int(p[1]))
            cv2.circle(image, center, fixed_size, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # Hold the current image and set it to the label
        self.current_image = image  
        self.set_image_to_label(image)

    def set_image_to_label(self, rgb_image):
        """
        Resize the image to fit the QLabel width while maintaining aspect ratio 
        and display it.
        """
        label_width = self.display_label.width()
        # Use a provisional value if the label width is 0 (window not displayed)
        if label_width == 0:
            label_width = 400

        h, w, _ = rgb_image.shape
        aspect_ratio = h / w

        new_width = label_width
        new_height = int(new_width * aspect_ratio)

        # Resize the image
        resized_image = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to QPixmap and set to QLabel
        height, width, channel = resized_image.shape
        bytes_per_line = channel * width
        q_image = QImage(resized_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.display_label.setPixmap(pixmap)
        # Clear any text that was displayed
        self.display_label.setText("")

    def load_config_data(self, config_path):
        """Load configuration data from a YAML file."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return {}
