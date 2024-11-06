import numpy as np
from PyQt5.QtWidgets import QWidget, QGridLayout, QSplitter, QListWidget, QVBoxLayout, QApplication, QMenu
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QVector3D, QMatrix4x4  # 必要なモジュールのインポート
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from utils.logger import setup_logger
import json
import sys

# ログの設定
logger = setup_logger()

def rpy_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Roll, pitch, yaw を回転行列に変換する関数"""
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

def load_reconstruction(file_path: str):
    """JSONファイルから再構築データをロードする関数"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.error(f"File {file_path} not found.")
        return None
    except json.JSONDecodeError:
        logger.error(f"File {file_path} is not valid JSON.")
        return None

class PointCloudVisualizer(QWidget):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

        # レイアウトの設定
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # GLViewWidgetの初期設定
        self.viewer = gl.GLViewWidget()
        self.viewer.setWindowTitle("Point Cloud and Camera Visualization")
        self.viewer.setCameraPosition(distance=50)
        self.layout.addWidget(self.viewer, 0, 0, 1, 1)

        self.camera_items = []
        
        # 初期表示の更新
        self.update_visualization()

    def update_visualization(self):
        """ポイントクラウドとカメラの可視化を更新"""
        self.viewer.items.clear()
        
        # 現在のファイルを取得してデータを読み込む
        data = load_reconstruction(self.file_path)
        if data is None:
            logger.error("No data loaded; cannot update visualization.")
            return

        # ポイントクラウドとカメラのデータを取得
        points, colors, cameras = [], [], []
        for reconstruction in data:
            if "points" not in reconstruction or "shots" not in reconstruction:
                logger.warning("Data format is incorrect; missing 'points' or 'shots'")
                continue

            points = np.array([p["coordinates"] for p in reconstruction["points"].values()], dtype=np.float32)
            colors = np.array([p["color"] for p in reconstruction["points"].values()], dtype=np.float32) / 255.0

            # データが空の場合を確認
            if points.size == 0 or colors.size == 0:
                print("No points or colors in data.")
                logger.warning("No points or colors in data.")
                continue

            # カメラデータの設定
            for shot_name, shot in reconstruction["shots"].items():
                rotation = shot.get("rotation", [0, 0, 0])
                roll, pitch, yaw = rotation
                R = rpy_to_rotmat(roll, pitch, yaw)
                t = np.array(shot.get("translation", [0, 0, 0]))
                t_rot = t[[0, 2, 1]] * np.array([-1, -1, 1])
                cam_type = reconstruction["cameras"][shot["camera"]]["projection_type"]
                cameras.append((shot_name, R, t_rot, cam_type))

        # ポイントクラウドの表示
        scatter = gl.GLScatterPlotItem(pos=points, color=colors, size=2)
        scatter.setGLOptions('translucent')  # エラーハンドリングとしての描画設定
        self.viewer.addItem(scatter)

        # カメラの可視化
        for shot_name, R, t, cam_model in cameras:
            self.add_camera_visualization(shot_name, R, t, cam_model)

    def add_camera_visualization(self, cam_name, R, t, cam_model, size=1):
        """カメラの位置と方向を可視化し、リストに保持"""
        if cam_model in ["spherical", "equirectangular"]:
            # 球状のカメラモデルの可視化
            sphere_meshdata = pg.opengl.MeshData.sphere(rows=10, cols=20, radius=size)
            sphere = gl.GLMeshItem(meshdata=sphere_meshdata, color=(1, 1, 1, 0.3), smooth=True, shader='balloon')
            sphere.setGLOptions('translucent')
            sphere.translate(t[0], t[1], t[2])
            self.viewer.addItem(sphere)
            self.camera_items.append((cam_name, sphere, t, R))

    def highlight_camera(self, cam_name, highlight_color=(1, 0, 0, 0.8)):
        """指定されたカメラをハイライト"""
        for name, sphere, _, _ in self.camera_items:
            if name == cam_name:
                sphere.setColor(highlight_color)
            else:
                sphere.setColor((1, 1, 1, 0.3))  # デフォルトカラーに戻す

    def move_to_camera(self, cam_name):
        """指定されたカメラの位置と回転に視点を移動"""
        for name, _, position, rotation_matrix in self.camera_items:
            if name == cam_name:
                # カメラ位置を新しい視点位置として設定
                self.viewer.setCameraPosition(pos=QVector3D(position[0], position[1], position[2]), distance=10)
                
                # 回転行列から視点のピッチとヨーを計算
                yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                pitch = np.arcsin(-rotation_matrix[2, 0])
                
                # 視点の回転を設定
                self.viewer.setCameraPosition(elevation=np.degrees(pitch), azimuth=np.degrees(yaw))
                break

class MainWindow(QWidget):
    def __init__(self, file_path):
        super().__init__()
        self.setWindowTitle("SAM2 Application")
        self.setGeometry(100, 100, 1200, 800)
        
        # スプリッターの作成
        splitter = QSplitter(Qt.Horizontal)
        
        # 左側: 画像リスト
        self.image_list = QListWidget()
        self.image_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_list.customContextMenuRequested.connect(self.open_context_menu)
        self.image_list.itemClicked.connect(self.on_image_click)
        
        # ポイントクラウドビジュアライザーのインスタンス作成
        self.pointcloud_viewer = PointCloudVisualizer(file_path)

        # レイアウトの設定
        splitter.addWidget(self.image_list)
        splitter.addWidget(self.pointcloud_viewer)
        splitter.setSizes([200, 800])

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        # カメラ名リストの設定
        camera_names = [name for name, _, _, _ in self.pointcloud_viewer.camera_items]
        self.populate_image_list(camera_names)

    def populate_image_list(self, camera_names):
        """画像リストをカメラ名で埋める"""
        self.image_list.clear()
        for name in camera_names:
            self.image_list.addItem(name)

    def open_context_menu(self, position: QPoint):
        """右クリックでコンテキストメニューを表示し、選択したカメラ位置に移動"""
        item = self.image_list.itemAt(position)
        if item:
            camera_name = item.text()
            menu = QMenu()
            move_action = menu.addAction("move to camera position")
            action = menu.exec_(self.image_list.mapToGlobal(position))
            if action == move_action:
                self.pointcloud_viewer.move_to_camera(camera_name)

    def on_image_click(self, item):
        """画像リストの項目がクリックされたときに対応するカメラをハイライト"""
        camera_name = item.text()
        self.pointcloud_viewer.highlight_camera(camera_name)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    file_path = "path/to/your/reconstruction.json"  # JSONファイルのパスを設定
    window = MainWindow(file_path)
    window.show()
    sys.exit(app.exec_())
