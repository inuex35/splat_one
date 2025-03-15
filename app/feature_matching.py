import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QSizePolicy,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt
from opensfm import dataset, matching, features 
from opensfm.actions import match_features
import yaml


class MatchingConfigDialog(QDialog):
    """Feature Matching Configuration Dialog"""
    def __init__(self, config_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Matching Configuration")
        self.config_path = config_path

        # Load configuration data
        with open(config_path, "r") as f:
            self.config_data = yaml.safe_load(f)

        # Set main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.fields = {}

        # Basic matching settings
        self.add_combobox_field("matching_type", ["Brute-Force", "FLANN", "LIGHTGLUE"], default="Brute-Force")
        self.add_lineedit_field("distance_threshold", field_type=float, default_value=0.75)
        self.add_checkbox_field("cross_check", default=False)

        # GPS distance-based settings
        self.add_lineedit_field("matching_gps_distance", field_type=int, default_value=150)
        self.add_lineedit_field("matching_gps_neighbors", field_type=int, default_value=0)
        
        # Time and name-based matching settings
        self.add_lineedit_field("matching_time_neighbors", field_type=int, default_value=0)
        self.add_lineedit_field("matching_order_neighbors", field_type=int, default_value=0)
        
        # BoW (Bag of Words) based settings
        self.add_lineedit_field("matching_bow_neighbors", field_type=int, default_value=0)
        self.add_lineedit_field("matching_bow_gps_distance", field_type=int, default_value=0)
        self.add_lineedit_field("matching_bow_gps_neighbors", field_type=int, default_value=0)
        self.add_checkbox_field("matching_bow_other_cameras", default=False)
        
        # VLAD based settings
        self.add_lineedit_field("matching_vlad_neighbors", field_type=int, default_value=0)
        self.add_lineedit_field("matching_vlad_gps_distance", field_type=int, default_value=0)
        self.add_lineedit_field("matching_vlad_gps_neighbors", field_type=int, default_value=0)
        self.add_checkbox_field("matching_vlad_other_cameras", default=False)
        
        # Other matching settings
        self.add_lineedit_field("matching_graph_rounds", field_type=int, default_value=0)
        self.add_checkbox_field("matching_use_filters", default=False)
        self.add_combobox_field("matching_use_segmentation", ["no", "yes"], default="no")

        # OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.save_config)
        button_box.rejected.connect(self.reject)
        self.main_layout.addWidget(button_box)

    def add_combobox_field(self, key, options, default=None):
        """Add combobox field"""
        label = QLabel(key)
        label.setStyleSheet("font-size: 14px;")  # Adjust as needed
        field = QComboBox()
        field.addItems(options)
        # Ensure the value is a string before passing to setCurrentText
        value = self.config_data.get(key, default)
        if not isinstance(value, str):
            value = str(value) if value is not None else default
        field.setCurrentText(value)
        
        row_layout = QHBoxLayout()
        row_layout.addWidget(label)
        row_layout.addWidget(field)
        self.main_layout.addLayout(row_layout)
        self.fields[key] = field

    def add_lineedit_field(self, key, field_type=str, default_value=None):
        """Add line edit field (numeric or string)"""
        label = QLabel(key)
        label.setStyleSheet("font-size: 14px;")  # Adjust as needed
        field = QLineEdit(str(self.config_data.get(key, default_value)))
        if field_type == int:
            field.setValidator(QIntValidator())
        elif field_type == float:
            field.setValidator(QDoubleValidator())
        row_layout = QHBoxLayout()
        row_layout.addWidget(label)
        row_layout.addWidget(field)
        self.main_layout.addLayout(row_layout)
        self.fields[key] = field

    def add_checkbox_field(self, key, default=False):
        """Add checkbox field"""
        label = QLabel(key)
        label.setStyleSheet("font-size: 14px;")  # Adjust as needed
        field = QCheckBox()
        field.setChecked(self.config_data.get(key, default))
        row_layout = QHBoxLayout()
        row_layout.addWidget(label)
        row_layout.addWidget(field)
        self.main_layout.addLayout(row_layout)
        self.fields[key] = field

    def save_config(self):
        """Save configuration and close dialog"""
        for key, field in self.fields.items():
            if isinstance(field, QCheckBox):
                self.config_data[key] = field.isChecked()
            elif isinstance(field, QLineEdit):
                text = field.text().strip()
                if text.replace('.', '', 1).isdigit():
                    if '.' in text:
                        self.config_data[key] = float(text)
                    else:
                        self.config_data[key] = int(text)
                else:
                    self.config_data[key] = text
            elif isinstance(field, QComboBox):
                self.config_data[key] = field.currentText()

        with open(self.config_path, "w") as f:
            yaml.safe_dump(self.config_data, f)
        QMessageBox.information(self, "Config", "Configuration saved successfully.")
        self.accept()

class FeatureMatching(QWidget):
    def __init__(self, workdir, image_list):
        super().__init__()
        self.workdir = workdir
        self.image_list = image_list
        self.dataset = dataset.DataSet(workdir)
        self.config_path = os.path.join(workdir, "config.yaml")

        # Hold left and right images (RGB)
        self.current_image_left = None
        self.current_image_right = None
        # Left and right image names
        self.current_image_name_left = None
        self.current_image_name_right = None

        # --- Create layout ---
        layout = QVBoxLayout()
        image_layout = QHBoxLayout()

        # --- Create left image view ---
        left_image_layout = QVBoxLayout()
        self.left_image_labels = self.create_image_viewer(side="Left")
        for label in self.left_image_labels[:2]:
            left_image_layout.addWidget(label)
        image_layout.addLayout(left_image_layout)

        # --- Create right image view ---
        right_image_layout = QVBoxLayout()
        self.right_image_labels = self.create_image_viewer(side="Right")
        for label in self.right_image_labels[:2]:
            right_image_layout.addWidget(label)
        image_layout.addLayout(right_image_layout)

        layout.addLayout(image_layout)

        # --- Bottom: Label for displaying matching results ---
        self.matching_label = QLabel("No matched images to show")
        self.matching_label.setAlignment(Qt.AlignCenter)
        self.matching_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Example: Gray text + larger font
        self.matching_label.setStyleSheet("color: gray; font-size: 16px;")

        layout.addWidget(self.matching_label)

        # --- Button layout ---
        button_layout = QHBoxLayout()

        self.match_button = QPushButton("Match Features")
        self.match_button.clicked.connect(self.run_match_features)
        button_layout.addWidget(self.match_button)

        self.config_button = QPushButton("Config")
        self.config_button.clicked.connect(self.configure_matching)
        button_layout.addWidget(self.config_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def create_image_viewer(self, side):
        """
        Receives side=="Left" or "Right",
        and creates two labels: one for displaying the original image and one for displaying keypoints.
        """
        labels = []
        # Label for original image
        label_original = QLabel(f"Select an image to view ({side} Original)")
        label_original.setAlignment(Qt.AlignCenter)
        label_original.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Example:
        label_original.setStyleSheet("color: gray; font-size: 16px;")

        # Label for displaying keypoints
        label_keypoints = QLabel(f"Select an image to view ({side} Keypoints)")
        label_keypoints.setAlignment(Qt.AlignCenter)
        label_keypoints.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Example:
        label_keypoints.setStyleSheet("color: gray; font-size: 16px;")

        labels.append(label_original)
        labels.append(label_keypoints)

        return labels

    def resizeEvent(self, event):
        """Resize event"""
        self.update_image_display("left")
        self.update_image_display("right")

    def load_image_by_name(self, image_name, position="left"):
        """Load image and set to left or right"""
        image_path = os.path.join(self.workdir, "images", image_name)
        if not os.path.exists(image_path):
            QMessageBox.warning(self, "Error", f"Image {image_name} not found.")
            return

        image = cv2.imread(image_path)
        if image is not None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if position == "left":
                self.current_image_left = rgb_image
                self.current_image_name_left = image_name
            else:
                self.current_image_right = rgb_image
                self.current_image_name_right = image_name

            # Update original image display
            self.update_image_display(position)
            # If both images are loaded, update keypoints and matches
            self.update_keypoints_and_matches()
        else:
            QMessageBox.warning(self, "Error", f"Failed to load image {image_name}.")

    def update_keypoints_and_matches(self):
        """Display keypoints and matches if both images are available"""
        if self.current_image_name_left and self.current_image_name_right:
            try:
                features_left = self.dataset.load_features(self.current_image_name_left)
            except FileNotFoundError:
                self.left_image_labels[1].setText("No Left Keypoints.")
                features_left = None
                
            try:
                features_right = self.dataset.load_features(self.current_image_name_right)
            except FileNotFoundError:
                self.right_image_labels[1].setText("No Right Keypoints.")
                features_right = None
            if features_left is None or features_right is None:
                return

            # Get image size and convert to pixel coordinates
            h_l, w_l = self.current_image_left.shape[:2]
            h_r, w_r = self.current_image_right.shape[:2]
            pixels_left = features.denormalized_image_coordinates(features_left.points, w_l, h_l)
            pixels_right = features.denormalized_image_coordinates(features_right.points, w_r, h_r)

            # Draw keypoints
            self.plot_keypoints(pixels_left, "left")
            self.plot_keypoints(pixels_right, "right")

            # Load matching results
            matches = self.dataset.find_matches(self.current_image_name_left, self.current_image_name_right)
            if len(matches) > 0:
                # Draw matching results
                points_left = features_left.points[matches[:, 0]]
                points_right = features_right.points[matches[:, 1]]
                self.plot_matches(points_left, points_right)
            else:
                # If no matches, draw keypoints side by side
                self.plot_combined_keypoints(pixels_left, pixels_right)

    def run_match_features(self):
        """Run feature detection on all images in the dataset."""
        match_features.run_dataset(self.dataset)
        QMessageBox.information(self, "Feature Matching", "Feature Matching completed for all images.")


    def configure_matching(self):
        """Matching configuration"""
        dialog = MatchingConfigDialog(config_path=self.config_path, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            QMessageBox.information(self, "Config", "Configuration saved successfully.")
            self.dataset = dataset.DataSet(self.workdir)

    def plot_keypoints(self, points, position):
        """Draw keypoints on image"""
        if position == "left" and self.current_image_left is not None:
            image = self.current_image_left.copy()
            for p in points:
                cv2.circle(image, (int(p[0]), int(p[1])), 5, (0, 255, 255), 2, cv2.LINE_AA)
            self.set_image_to_label(image, "keypoints", "left")
        elif position == "right" and self.current_image_right is not None:
            image = self.current_image_right.copy()
            for p in points:
                cv2.circle(image, (int(p[0]), int(p[1])), 5, (0, 255, 255), 2, cv2.LINE_AA)
            self.set_image_to_label(image, "keypoints", "right")

    def plot_matches(self, points_left, points_right):
        """Display matching results of left and right images combined into one"""
        h1, w1, _ = self.current_image_left.shape
        h2, w2, _ = self.current_image_right.shape
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = self.current_image_left
        canvas[:h2, w1:] = self.current_image_right

        # Convert coordinates to screen size
        points_left = features.denormalized_image_coordinates(points_left, w1, h1)
        points_right = features.denormalized_image_coordinates(points_right, w2, h2)

        for (x1, y1), (x2, y2) in zip(points_left, points_right):
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2) + w1, int(y2))
            cv2.circle(canvas, pt1, 5, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, pt2, 5, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, pt1, pt2, (255, 255, 0), 1)

        self.set_image_to_label(canvas, "matches")

    def plot_combined_keypoints(self, points_left, points_right):
        """Draw keypoints of left and right images side by side if no matches"""
        h1, w1, _ = self.current_image_left.shape
        h2, w2, _ = self.current_image_right.shape
        combined = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        left_img = self.current_image_left.copy()
        right_img = self.current_image_right.copy()

        for (x1, y1) in points_left:
            cv2.circle(left_img, (int(x1), int(y1)), 5, (0, 255, 255), 2, cv2.LINE_AA)
        for (x2, y2) in points_right:
            cv2.circle(right_img, (int(x2), int(y2)), 5, (0, 255, 255), 2, cv2.LINE_AA)

        combined[:h1, :w1] = left_img
        combined[:h2, w1:] = right_img
        self.set_image_to_label(combined, "matches")

    def set_image_to_label(self, rgb_image, label_type, position=None):
        """Resize and display image on label"""
        if label_type == "matches":
            label = self.matching_label
        else:
            if position == "left":
                label = self.left_image_labels[0] if label_type == "original" else self.left_image_labels[1]
            else:
                label = self.right_image_labels[0] if label_type == "original" else self.right_image_labels[1]

        label_width = label.width()
        if label_width == 0:
            label_width = 300

        h, w, _ = rgb_image.shape
        aspect_ratio = h / w
        new_width = label_width
        new_height = int(new_width * aspect_ratio)

        resized_image = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        height, width, channel = resized_image.shape
        bytes_per_line = channel * width
        q_image = QImage(resized_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)

        # Clear text if any
        label.setText("")

    def update_image_display(self, position):
        """Redraw original image"""
        if position == "left" and self.current_image_left is not None:
            self.set_image_to_label(self.current_image_left, "original", "left")
        elif position == "right" and self.current_image_right is not None:
            self.set_image_to_label(self.current_image_right, "original", "right")
