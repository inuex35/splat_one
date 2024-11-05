# main_app.py
import os
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QToolBar, QAction, QFileDialog, QLabel, QVBoxLayout, QWidget, QMessageBox, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint
import cv2
from sam2mask.prompt_gui import PromptGUI  # Import PromptGUI class

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM2 Mask Creator")
        self.setGeometry(100, 100, 1000, 800)
        
        # Set up toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Load Workdir action
        load_workdir_action = QAction("Load Workdir", self)
        load_workdir_action.triggered.connect(self.load_workdir)
        toolbar.addAction(load_workdir_action)

        # Initialize PromptGUI
        checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
        config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.prompt_gui = PromptGUI(checkpoint_path, config_path)

        # Initialization
        self.workdir = None
        self.img_dir = None
        self.mask_dir = None
        self.input_points = []  # Coordinates of clicked points (cleared per image)
        self.input_labels = []  # Labels for each point (cleared per image)
        self.label_toggle = 1  # Toggle label between 1 and 0

        # Image list and index
        self.image_list = []
        self.current_image_index = 0  # Index of the currently displayed image
        self.current_mask = None  # Initialize to store the combined mask

        # Label for displaying images
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        # Navigation buttons
        self.prev_button = QPushButton("< Previous Image", self)
        self.prev_button.clicked.connect(self.prev_image)
        layout.addWidget(self.prev_button, alignment=Qt.AlignLeft)

        self.next_button = QPushButton("Next Image >", self)
        self.next_button.clicked.connect(self.next_image)
        layout.addWidget(self.next_button, alignment=Qt.AlignRight)

        # Set main widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_workdir(self):
        # Display dialog to select workdir
        self.workdir = QFileDialog.getExistingDirectory(self, "Select Workdir")
        if not self.workdir:
            return

        # Check for images folder
        self.img_dir = os.path.join(self.workdir, "images")
        if not os.path.exists(self.img_dir):
            QMessageBox.warning(self, "Error", "images folder does not exist.")
            self.workdir = None
        else:
            # Create masks folder
            self.mask_dir = os.path.join(self.workdir, "masks")
            os.makedirs(self.mask_dir, exist_ok=True)
            self.image_list = [f for f in os.listdir(self.img_dir) if f.endswith(".png") or f.endswith(".jpg")]
            if self.image_list:
                self.display_current_image()
            QMessageBox.information(self, "Success", "Workdir loaded successfully.")

    def display_current_image(self):
        """Display the image at the current index"""
        if self.image_list and 0 <= self.current_image_index < len(self.image_list):
            image_path = os.path.join(self.img_dir, self.image_list[self.current_image_index])
            self.current_image = cv2.imread(image_path)
            self.show_image_on_label(self.current_image)

            # Clear coordinates, labels, and initialize the mask to white
            self.input_points.clear()
            self.input_labels.clear()
            self.current_mask = np.ones(self.current_image.shape[:2], dtype=np.uint8) * 255  # White mask

            # Save the initial white mask
            self.save_current_mask()

    def show_image_on_label(self, image):
        """Display the image on QLabel, resized to fit the current window size"""
        # Convert BGR to RGB for QImage
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Create a QPixmap from the QImage
        pixmap = QPixmap.fromImage(q_image)

        # Resize pixmap to fit QLabel's size
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def generate_mask(self):
        if not self.img_dir or not os.path.exists(self.img_dir):
            QMessageBox.warning(self, "Error", "Please load a valid workdir first.")
            return

        # クリックされた点の座標とラベルでマスクを生成
        new_mask = self.prompt_gui.process_single_image(
            self.current_image,
            self.mask_dir,
            self.image_list[self.current_image_index],
            point_coords=np.array(self.input_points),
            point_labels=np.array(self.input_labels),
        )

        # new_mask が None の場合のエラーチェック
        if new_mask is None:
            QMessageBox.warning(self, "Error", "Failed to generate mask. Check SAM model or input parameters.")
            return

        # 初めてマスクを生成する場合、self.current_mask を初期化
        if self.current_mask is None:
            self.current_mask = new_mask
        else:
            # self.current_mask がすでにある場合は、新しいマスクと組み合わせる
            if new_mask.shape != self.current_mask.shape:
                new_mask = cv2.resize(new_mask, (self.current_mask.shape[1], self.current_mask.shape[0]))
            self.current_mask = cv2.bitwise_or(self.current_mask, new_mask)

        # アルファブレンドで画像とマスクを重ねて表示
        self.display_mask_image(self.current_mask)
        self.save_current_mask()

    def display_mask_image(self, mask):
        """Display the image with the mask overlaid using alpha blending"""
        if mask is not None:
            # Expand mask to 3 channels and blend
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            blended_image = cv2.addWeighted(self.current_image, 0.7, mask_3ch, 0.3, 0)
            self.show_image_on_label(blended_image)
        else:
            QMessageBox.warning(self, "Error", "Mask image not found.")

    def save_current_mask(self):
        """Save the current mask to the mask directory"""
        mask_output_path = os.path.join(self.mask_dir, f"mask_{self.image_list[self.current_image_index]}")
        cv2.imwrite(mask_output_path, self.current_mask)
        print(f"Mask saved to {mask_output_path}")

    def mousePressEvent(self, event):
        """Handle click events to add prompts to the current image and automatically call mask generation"""
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            if self.image_label.geometry().contains(pos):
                label_pos = self.image_label.mapFromParent(pos)
                self.input_points.append([label_pos.x(), label_pos.y()])
                self.input_labels.append(self.label_toggle)
                
                # Toggle label
                self.label_toggle = 1 - self.label_toggle

                # Call mask generation immediately after a point is added
                self.generate_mask()

    def next_image(self):
        """Move to the next image"""
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.display_current_image()

    def prev_image(self):
        """Move to the previous image"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_current_image()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
