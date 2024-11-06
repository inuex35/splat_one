import os
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QToolBar, QAction, QFileDialog, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QMessageBox, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
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

        # Navigation and Reset Buttons Layout
        button_layout = QHBoxLayout()
        
        # Previous Image Button
        self.prev_button = QPushButton("< Previous Image", self)
        self.prev_button.clicked.connect(self.prev_image)
        button_layout.addWidget(self.prev_button, alignment=Qt.AlignLeft)

        # Reset Mask Button
        self.reset_button = QPushButton("Reset Mask", self)
        self.reset_button.clicked.connect(self.reset_mask)
        button_layout.addWidget(self.reset_button, alignment=Qt.AlignCenter)

        # Next Image Button
        self.next_button = QPushButton("Next Image >", self)
        self.next_button.clicked.connect(self.next_image)
        button_layout.addWidget(self.next_button, alignment=Qt.AlignRight)

        layout.addLayout(button_layout)

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

            # マスクの読み込み
            mask_path = os.path.join(self.mask_dir, f"mask_{self.image_list[self.current_image_index]}")
            if os.path.exists(mask_path):
                self.current_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                print(f"Loaded existing mask from {mask_path}")
            else:
                # マスクがない場合は白のマスクで初期化
                self.current_mask = np.ones(self.current_image.shape[:2], dtype=np.uint8) * 255

            # 座標リストとラベルをクリア
            self.input_points.clear()
            self.input_labels.clear()

            # 現在のマスクを表示
            self.display_mask_image(self.current_mask)
            print(f"Displayed mask for {self.image_list[self.current_image_index]}")

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

    def reset_mask(self):
        """Reset the current mask to a blank white mask and update the display"""
        if self.current_image is None:
            QMessageBox.warning(self, "Error", "No image loaded to reset the mask.")
            return

        # 白いマスクで初期化
        self.current_mask = np.ones(self.current_image.shape[:2], dtype=np.uint8) * 255

        # マスクファイルを上書き保存
        mask_output_path = os.path.join(self.mask_dir, f"mask_{self.image_list[self.current_image_index]}")
        cv2.imwrite(mask_output_path, self.current_mask)
        print(f"Mask reset and saved to {mask_output_path}")

        # 画像とリセットされたマスクを再表示
        self.display_mask_image(self.current_mask)
        QMessageBox.information(self, "Reset Mask", "The mask has been reset to a blank state.")


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
            # サイズが同じか確認し、異なる場合はnew_maskをself.current_maskのサイズにリサイズ
            if new_mask.shape != self.current_mask.shape:
                new_mask = cv2.resize(new_mask, (self.current_mask.shape[1], self.current_mask.shape[0]))

            # データ型が異なる場合は、new_maskをself.current_maskの型に変換
            if new_mask.dtype != self.current_mask.dtype:
                new_mask = new_mask.astype(self.current_mask.dtype)

        # マスクの組み合わせ
        self.current_mask = cv2.bitwise_and(self.current_mask, new_mask)

        # アルファブレンドで画像とマスクを重ねて表示
        self.display_mask_image(self.current_mask)
        self.save_current_mask()

    def display_mask_image(self, mask):
        """Display the image with the mask overlaid using alpha blending on black regions only"""
        if mask is not None:
            # マスクを3チャンネルに変換
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # マスクの黒い部分のみを抽出
            black_region = cv2.inRange(mask_3ch, (0, 0, 0), (0, 0, 0))
            black_region_3ch = cv2.cvtColor(black_region, cv2.COLOR_GRAY2BGR) / 255.0

            # 黒い部分のみをブレンド
            blended_image = (self.current_image * (1 - black_region_3ch) +
                             mask_3ch * black_region_3ch * 0.7 + 
                             self.current_image * black_region_3ch * 0.3).astype(np.uint8)
            
            # 画像を表示
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
            # グローバルなクリック位置からQLabelの位置に変換
            label_pos = self.image_label.mapFromGlobal(event.globalPos())

            # QLabel内で画像が表示されている範囲のサイズを取得
            if self.image_label.pixmap():
                pixmap_rect = self.image_label.pixmap().rect()
                pixmap_width, pixmap_height = pixmap_rect.width(), pixmap_rect.height()

                # QLabelのサイズとpixmapのサイズを比較し、中央配置の余白を計算
                label_width, label_height = self.image_label.width(), self.image_label.height()
                offset_x = (label_width - pixmap_width) / 2
                offset_y = (label_height - pixmap_height) / 2

                # クリック位置が表示範囲内か確認
                if (offset_x <= label_pos.x() <= offset_x + pixmap_width) and (offset_y <= label_pos.y() <= offset_y + pixmap_height):
                    # クリック位置をオリジナルの画像座標に変換
                    scale_x = self.current_image.shape[1] / pixmap_width
                    scale_y = self.current_image.shape[0] / pixmap_height
                    corrected_x = int((label_pos.x() - offset_x) * scale_x)
                    corrected_y = int((label_pos.y() - offset_y) * scale_y)

                    # 画像範囲内に制限
                    corrected_x = max(0, min(corrected_x, self.current_image.shape[1] - 1))
                    corrected_y = max(0, min(corrected_y, self.current_image.shape[0] - 1))

                    # 座標を追加
                    self.input_points.append([corrected_x, corrected_y])
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
