import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QSizePolicy, QCheckBox, QComboBox, QDialogButtonBox, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt
from opensfm import dataset, features
from itertools import combinations
import yaml

class MatchingConfigDialog(QDialog):
    """特徴点マッチングの設定ダイアログ"""
    def __init__(self, config_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Matching Configuration")
        self.config_path = config_path

        # 設定データを読み込む
        with open(config_path, "r") as f:
            self.config_data = yaml.safe_load(f)

        # メインレイアウトの設定
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.fields = {}

        # 基本的なマッチング設定
        self.add_combobox_field("matching_type", ["Brute-Force", "FLANN", "LIGHTGLUE"], default="Brute-Force")
        self.add_lineedit_field("distance_threshold", field_type=float, default_value=0.75)
        self.add_checkbox_field("cross_check", default=False)

        # GPS距離に基づく設定
        self.add_lineedit_field("matching_gps_distance", field_type=int, default_value=150)
        self.add_lineedit_field("matching_gps_neighbors", field_type=int, default_value=0)
        
        # 時間・名前ベースのマッチング設定
        self.add_lineedit_field("matching_time_neighbors", field_type=int, default_value=0)
        self.add_lineedit_field("matching_order_neighbors", field_type=int, default_value=0)
        
        # BoW（Bag of Words）ベースの設定
        self.add_lineedit_field("matching_bow_neighbors", field_type=int, default_value=0)
        self.add_lineedit_field("matching_bow_gps_distance", field_type=int, default_value=0)
        self.add_lineedit_field("matching_bow_gps_neighbors", field_type=int, default_value=0)
        self.add_checkbox_field("matching_bow_other_cameras", default=False)
        
        # VLADベースの設定
        self.add_lineedit_field("matching_vlad_neighbors", field_type=int, default_value=0)
        self.add_lineedit_field("matching_vlad_gps_distance", field_type=int, default_value=0)
        self.add_lineedit_field("matching_vlad_gps_neighbors", field_type=int, default_value=0)
        self.add_checkbox_field("matching_vlad_other_cameras", default=False)
        
        # その他のマッチング設定
        self.add_lineedit_field("matching_graph_rounds", field_type=int, default_value=0)
        self.add_checkbox_field("matching_use_filters", default=False)
        self.add_combobox_field("matching_use_segmentation", ["no", "yes"], default="no")

        # OK と Cancel ボタン
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.save_config)
        button_box.rejected.connect(self.reject)
        self.main_layout.addWidget(button_box)

    def add_combobox_field(self, key, options, default=None):
        """コンボボックスフィールドを追加"""
        label = QLabel(key)
        field = QComboBox()
        field.addItems(options)
        # 文字列型であることを確認して setCurrentText に渡す
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
        """テキストフィールド（数値や文字列）を追加"""
        label = QLabel(key)
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
        """チェックボックスフィールドを追加"""
        label = QLabel(key)
        field = QCheckBox()
        field.setChecked(self.config_data.get(key, default))
        row_layout = QHBoxLayout()
        row_layout.addWidget(label)
        row_layout.addWidget(field)
        self.main_layout.addLayout(row_layout)
        self.fields[key] = field

    def save_config(self):
        """設定を保存してダイアログを閉じる"""
        for key, field in self.fields.items():
            if isinstance(field, QCheckBox):
                self.config_data[key] = field.isChecked()
            elif isinstance(field, QLineEdit):
                text = field.text()
                self.config_data[key] = float(text) if '.' in text else int(text) if text.isdigit() else text
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
        self.current_image_left = None
        self.current_image_right = None
        self.current_image_name_left = None
        self.current_image_name_right = None

        # UI のセットアップ
        layout = QVBoxLayout()
        image_layout = QHBoxLayout()

        # 左画像の2つのラベル
        left_image_layout = QVBoxLayout()
        self.left_image_labels = self.create_image_viewer("Left Image")
        for label in self.left_image_labels[:2]:
            left_image_layout.addWidget(label)
        image_layout.addLayout(left_image_layout)

        # 右画像の2つのラベル
        right_image_layout = QVBoxLayout()
        self.right_image_labels = self.create_image_viewer("Right Image")
        for label in self.right_image_labels[:2]:
            right_image_layout.addWidget(label)
        image_layout.addLayout(right_image_layout)

        layout.addLayout(image_layout)

        # 最下段：マッチング結果表示用ラベル
        self.matching_label = QLabel("Matching Image")
        self.matching_label.setAlignment(Qt.AlignCenter)
        self.matching_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.matching_label)

        button_layout = QHBoxLayout()

        # マッチング実行ボタン
        self.match_button = QPushButton("Match Features")
        self.match_button.clicked.connect(self.run_match_features)
        button_layout.addWidget(self.match_button)

        # Reset Mask Button
        self.config_button = QPushButton("Config")
        self.config_button.clicked.connect(self.configure_matching)
        button_layout.addWidget(self.config_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def create_image_viewer(self, title):
        """画像、特徴点を表示する2つのラベルを返す"""
        labels = []
        for _ in range(2):
            label = QLabel(title)
            label.setAlignment(Qt.AlignCenter)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            labels.append(label)
        return labels

    def resizeEvent(self, event):
        """リサイズイベントを処理し、表示中の画像を QLabel のサイズに合わせて調整します。"""
        self.update_image_display("left")
        self.update_image_display("right")

    def load_image_by_name(self, image_name, position="left"):
        """指定された画像を読み込み、左右どちらかの位置に表示し、特徴点とマッチングも更新します。"""
        image_path = os.path.join(self.workdir, "images", image_name)
        image = cv2.imread(image_path)
        if image is not None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if position == "left":
                self.current_image_left = rgb_image
                self.current_image_name_left = image_name
            elif position == "right":
                self.current_image_right = rgb_image
                self.current_image_name_right = image_name
            self.update_image_display(position)

            # 特徴点とマッチングを更新
            self.update_keypoints_and_matches()
        else:
            QMessageBox.warning(self, "Error", f"Image {image_name} not found.")

    def update_keypoints_and_matches(self):
        """左右の画像が両方選択されている場合、特徴点とマッチング画像を更新します。"""
        if self.current_image_name_left and self.current_image_name_right:
            features_left = self.dataset.load_features(self.current_image_name_left)
            features_right = self.dataset.load_features(self.current_image_name_right)

            if features_left is None or features_right is None:
                QMessageBox.warning(self, "Error", "Feature data missing.")
                return

            h_l, w_l = self.current_image_left.shape[:2]
            h_r, w_r = self.current_image_right.shape[:2]
            pixels_left = features.denormalized_image_coordinates(features_left.points, w_l, h_l)
            pixels_right = features.denormalized_image_coordinates(features_right.points, w_r, h_r)

            # 特徴点の表示
            self.plot_keypoints(pixels_left, "left")
            self.plot_keypoints(pixels_right, "right")

            # マッチングを取得
            matches = self.dataset.find_matches(self.current_image_name_left, self.current_image_name_right)

            if len(matches) > 0:
                # マッチング結果のプロット
                points_left = features_left.points[matches[:, 0]]
                points_right = features_right.points[matches[:, 1]]
                self.plot_matches(points_left, points_right)
            else:
                # マッチング結果がない場合はキーポイント画像を連結して表示
                self.plot_combined_keypoints(pixels_left, pixels_right)

    def run_match_features(self):
        """現在選択されている左右の画像に対して特徴点マッチングを実行し、結果を表示します。"""
        if not self.current_image_name_left or not self.current_image_name_right:
            QMessageBox.warning(self, "Error", "Both left and right images must be loaded for matching.")
            return

        # 特徴点とマッチングを更新
        self.update_keypoints_and_matches()

    def configure_matching(self):
        """マッチング設定ダイアログを開く"""
        dialog = MatchingConfigDialog(config_path=self.config_path, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            QMessageBox.information(self, "Config", "Configuration saved successfully.")
            # 設定が更新されたため、必要に応じて dataset を再読み込みするなどの処理
            self.dataset = dataset.DataSet(self.workdir)

    def plot_keypoints(self, points, position):
        """元画像に特徴点をプロットして表示"""
        if position == "left" and self.current_image_left is not None:
            image = self.current_image_left.copy()
            for p in points:
                center = (int(p[0]), int(p[1]))
                cv2.circle(image, center, 5, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            self.set_image_to_label(image, "keypoints", position)
        elif position == "right" and self.current_image_right is not None:
            image = self.current_image_right.copy()
            for p in points:
                center = (int(p[0]), int(p[1]))
                cv2.circle(image, center, 5, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            self.set_image_to_label(image, "keypoints", position)

    def plot_matches(self, points_left, points_right):
        """左右の画像の特徴点マッチング結果を1枚の画像にプロットして表示"""
        h1, w1, _ = self.current_image_left.shape
        h2, w2, _ = self.current_image_right.shape
        output_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        output_image[:h1, :w1] = self.current_image_left
        output_image[:h2, w1:] = self.current_image_right
        points_left = features.denormalized_image_coordinates(points_left, w1, h1)
        points_right = features.denormalized_image_coordinates(points_right, w2, h2)
        for (x1, y1), (x2, y2) in zip(points_left, points_right):
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2) + w1, int(y2))  # 右画像はオフセットを加算
            cv2.circle(output_image, pt1, 5, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(output_image, pt2, 5, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.line(output_image, pt1, pt2, (255, 255, 0), 1)

        self.set_image_to_label(output_image, "matches")

    def plot_combined_keypoints(self, points_left, points_right):
        """左右のキーポイント画像を連結して1つの画像に表示"""
        h1, w1, _ = self.current_image_left.shape
        h2, w2, _ = self.current_image_right.shape
        combined_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        left_image = self.current_image_left.copy()
        right_image = self.current_image_right.copy()

        for (x1, y1) in points_left:
            cv2.circle(left_image, (int(x1), int(y1)), 5, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        for (x2, y2) in points_right:
            cv2.circle(right_image, (int(x2), int(y2)), 5, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        combined_image[:h1, :w1] = left_image
        combined_image[:h2, w1:] = right_image
        self.set_image_to_label(combined_image, "matches")
        
    def set_image_to_label(self, rgb_image, label_type, position=None):
        """指定された QLabel に画像をリサイズして表示"""
        if label_type == "matches":
            label = self.matching_label
        else:
            if position == "left":
                label = self.left_image_labels[0] if label_type == "original" else self.left_image_labels[1]
            else:
                label = self.right_image_labels[0] if label_type == "original" else self.right_image_labels[1]

        label_width = label.width()
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

    def update_image_display(self, position):
        """画像表示を更新"""
        if position == "left" and self.current_image_left is not None:
            self.set_image_to_label(self.current_image_left, "original", "left")
        elif position == "right" and self.current_image_right is not None:
            self.set_image_to_label(self.current_image_right, "original", "right")
