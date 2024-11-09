import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from opensfm import dataset
from opensfm.actions import match_features
from opensfm import matching

class FeatureMatching(QWidget):
    def __init__(self, workdir, image_list):
        super().__init__()
        self.workdir = workdir
        self.image_list = image_list
        self.dataset = dataset.DataSet(workdir)
        self.current_image_left = None
        self.current_image_right = None

        # UI のセットアップ
        main_layout = QVBoxLayout()

        # 上部に空間を追加して中央に配置
        main_layout.addStretch(1)

        # 左右の画像表示用ラベルを追加
        image_layout = QHBoxLayout()
        self.display_label_left = QLabel("Left Image")
        self.display_label_left.setAlignment(Qt.AlignCenter)
        self.display_label_left.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_layout.addWidget(self.display_label_left)

        self.display_label_right = QLabel("Right Image")
        self.display_label_right.setAlignment(Qt.AlignCenter)
        self.display_label_right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_layout.addWidget(self.display_label_right)

        main_layout.addLayout(image_layout)

        # 下部に空間を追加して中央に配置
        main_layout.addStretch(1)

        # 全画像のマッチング実行ボタン
        self.match_button = QPushButton("Match Features for All Images")
        self.match_button.clicked.connect(self.run_match_features)
        main_layout.addWidget(self.match_button, alignment=Qt.AlignBottom)

        self.setLayout(main_layout)

    def resizeEvent(self, event):
        """リサイズイベントを処理し、表示中の画像を QLabel のサイズに合わせて調整します。"""
        if self.current_image_left is not None:
            self.set_image_to_label(self.current_image_left, position="left")
        if self.current_image_right is not None:
            self.set_image_to_label(self.current_image_right, position="right")

    def load_image_by_name(self, image_name, position="left"):
        """指定された画像を読み込み、左右どちらかの位置に表示します。"""
        image_path = os.path.join(self.workdir, "images", image_name)
        image = cv2.imread(image_path)
        if image is not None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if position == "left":
                self.current_image_left = rgb_image
                self.set_image_to_label(rgb_image, position="left")
            elif position == "right":
                self.current_image_right = rgb_image
                self.set_image_to_label(rgb_image, position="right")
        else:
            QMessageBox.warning(self, "Error", f"Image {image_name} not found.")

    def run_match_features(self):
        """全画像での特徴点マッチングを実行します。"""
        match_features.run_dataset(self.dataset)
        QMessageBox.information(self, "Matching Completed", "Feature matching completed for all images.")

    def plot_matches(self, matches_data):
        """左右の画像の特徴点マッチング結果をプロットし、表示します。"""
        if self.current_image_left is None or self.current_image_right is None:
            QMessageBox.warning(self, "Error", "Both left and right images must be loaded.")
            return

        # 左右の画像のコピーを作成してマッチングラインを描画
        img_left = self.current_image_left.copy()
        img_right = self.current_image_right.copy()

        for (x1, y1), (x2, y2) in matches_data:
            cv2.circle(img_left, (int(x1), int(y1)), 5, (0, 255, 0), -1)
            cv2.circle(img_right, (int(x2), int(y2)), 5, (255, 0, 0), -1)

        # 描画後の画像をセット
        self.set_image_to_label(img_left, position="left")
        self.set_image_to_label(img_right, position="right")

    def set_image_to_label(self, rgb_image, position="left"):
        """QLabel に表示するため、画像をリサイズしてセットします。"""
        label = self.display_label_left if position == "left" else self.display_label_right
        label_width = label.width()
        h, w, _ = rgb_image.shape
        aspect_ratio = h / w

        # QLabel の幅に合わせてリサイズ
        new_width = label_width
        new_height = int(new_width * aspect_ratio)
        resized_image = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # QImage に変換して QLabel に設定
        height, width, channel = resized_image.shape
        bytes_per_line = channel * width
        q_image = QImage(resized_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)
