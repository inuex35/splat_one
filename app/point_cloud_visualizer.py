import numpy as np
from PyQt5.QtWidgets import QWidget, QGridLayout
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from utils.logger import setup_logger
import json

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
                cameras.append((R, t_rot, cam_type))

        # ポイントクラウドの表示
        scatter = gl.GLScatterPlotItem(pos=points, color=colors, size=2)
        scatter.setGLOptions('translucent')  # エラーハンドリングとしての描画設定
        self.viewer.addItem(scatter)

        # カメラの可視化
        for R, t, cam_model in cameras:
            self.add_camera_visualization(R, t, cam_model)

    def add_camera_visualization(self, R, t, cam_model, size=1):
        """カメラの位置と方向を可視化"""
        try:
            if cam_model in ["spherical", "equirectangular"]:
                # 球状のカメラモデルの可視化
                sphere_meshdata = pg.opengl.MeshData.sphere(rows=10, cols=20, radius=size)
                sphere = gl.GLMeshItem(meshdata=sphere_meshdata, color=(1, 1, 1, 0.3), smooth=True, shader='balloon')
                sphere.setGLOptions('translucent')
                sphere.translate(t[0], t[1], t[2])
                self.viewer.addItem(sphere)
            else:
                # 角柱カメラモデルの可視化
                corners = np.array([
                    [size, size, 0],
                    [size, -size, 0],
                    [-size, -size, 0],
                    [-size, size, 0],
                    [0, 0, -size * 2]
                ])
                corners = (R @ corners.T).T + t
                lines = np.array([
                    [corners[0], corners[1]],
                    [corners[1], corners[2]],
                    [corners[2], corners[3]],
                    [corners[3], corners[0]],
                    [corners[0], corners[4]],
                    [corners[1], corners[4]],
                    [corners[2], corners[4]],
                    [corners[3], corners[4]]
                ])
                for line in lines:
                    line_item = gl.GLLinePlotItem(pos=line, color=(0, 0, 1, 1), width=1)
                    self.viewer.addItem(line_item)
        except Exception as e:
            logger.error(f"Error while drawing camera visualization: {e}")
