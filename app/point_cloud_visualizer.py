import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGridLayout, QMessageBox
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QVector3D, QMatrix4x4  # 必要なモジュールのインポート
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from opensfm import dataset
from utils.logger import setup_logger
import json
import sys
from scipy.spatial.transform import Rotation
from opensfm.actions import reconstruct, create_tracks
from opensfm.reconstruction import ReconstructionAlgorithm
import os
# ログの設定
logger = setup_logger()

def rotation_vector_to_euler(rotation_vector):
    rotation_matrix = Rotation.from_rotvec(rotation_vector).as_matrix()
    return rotation_matrix

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

class Reconstruction(QWidget):
    def __init__(self, workdir):
        super().__init__()
        self.workdir = workdir
        self.reconstruction_file = os.path.join(workdir, "reconstruction.json")
        self.dataset = dataset.DataSet(workdir)
        self.camera_size = 1.0

        self.setFocusPolicy(Qt.StrongFocus)  # Ensure parent widget can receive keyboard events
        self.setFocus()

        # Set up main layout
        main_layout = QVBoxLayout(self)
        
        self.viewer = gl.GLViewWidget()
        self.viewer.setFocusPolicy(Qt.NoFocus)  # Prevent child widget from capturing key events
        self.viewer.setWindowTitle("Point Cloud and Camera Visualization")
        self.viewer.setCameraPosition(distance=50)
        main_layout.addWidget(self.viewer)
        
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Reconstruction")
        self.run_button.clicked.connect(self.run_reconstruction)
        button_layout.addWidget(self.run_button)

        self.config_button = QPushButton("Config")
        self.config_button.clicked.connect(self.configure_reconstruction)
        button_layout.addWidget(self.config_button)
        
        main_layout.addLayout(button_layout)
        
        # Initialize camera items list
        self.camera_items = []
        self.update_visualization()


    def run_reconstruction(self):
        """Run reconstruction process"""
        logger.info("Running reconstruction...")
        # Reconstruction logic here
        # For now, just update the visualization
        self.update_visualization()
        data = load_reconstruction(self.reconstruction_file)
        if data is not None:
            reply = QMessageBox.question(self, 'Confirmation', 'Reconstruction data found. Do you want to run the reconstruction?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                # Proceed with reconstruction logic
                logger.info("User confirmed to run reconstruction.")
                create_tracks.run_dataset(self.dataset)
                reconstruct.run_dataset(self.dataset, ReconstructionAlgorithm.INCREMENTAL) 
                self.update_visualization()   
            else:
                pass
        else:
            create_tracks.run_dataset(self.dataset)
            reconstruct.run_dataset(self.dataset, ReconstructionAlgorithm.INCREMENTAL)
            self.update_visualization()

    def configure_reconstruction(self):
        """Open the configuration dialog for feature extraction."""
        #QMessageBox.information(self, "Feature Extraction", "Feature extraction configuration dialog.")

    def keyPressEvent(self, event):
        # Increase camera size when '+' key is pressed
        if event.key() == Qt.Key_Plus:
            self.camera_size *= 1.1
            self.update_visualization()
        # Decrease camera size when '-' key is pressed
        elif event.key() == Qt.Key_Minus:
            self.camera_size /= 1.1
            self.update_visualization()
        else:
            super().keyPressEvent(event)

    def update_visualization(self):
        """Update the point cloud and camera visualization."""
        self.viewer.items.clear()
        data = load_reconstruction(self.reconstruction_file)
        if data is None:
            self.show_placeholder()
            return

        points, colors, cameras = [], [], []
        for reconstruction in data:
            if "points" not in reconstruction or "shots" not in reconstruction:
                logger.warning("Data format is incorrect; missing 'points' or 'shots'")
                continue

            points = np.array([p["coordinates"] for p in reconstruction["points"].values()], dtype=np.float32)
            colors = np.array([p["color"] for p in reconstruction["points"].values()], dtype=np.float32) / 255.0

            if points.size == 0 or colors.size == 0:
                logger.warning("No points or colors in data.")
                continue

            for shot_name, shot in reconstruction["shots"].items():
                try:
                    axis_angle = np.array(shot.get("rotation", [0, 0, 0]), dtype=np.float64)
                    rotation = Rotation.from_rotvec(axis_angle).as_matrix()
                    translation = np.array(shot.get("translation", [0, 0, 0]), dtype=np.float64)
                    position = -rotation.T @ translation
                    cam_type = reconstruction["cameras"][shot["camera"]]["projection_type"]
                    cameras.append((shot_name, rotation, position, cam_type))
                except Exception as e:
                    logger.error(f"Error processing shot '{shot_name}': {e}")
                    continue

        scatter = gl.GLScatterPlotItem(pos=points, color=colors, size=2)
        scatter.setGLOptions('translucent')
        self.viewer.addItem(scatter)

        for shot_name, R, t, cam_model in cameras:
            # Pass the dynamic camera size here
            self.add_camera_visualization(shot_name, R, t, cam_model, size=self.camera_size)

    def add_camera_visualization(self, cam_name, R, t, cam_model, size=1):
        """Visualize camera position and orientation, store in a list."""
        if cam_model in ["spherical", "equirectangular"]:
            sphere_meshdata = pg.opengl.MeshData.sphere(rows=10, cols=20, radius=size)
            sphere = gl.GLMeshItem(meshdata=sphere_meshdata, color=(1, 1, 1, 0.3), smooth=True, shader='balloon')
            sphere.setGLOptions('translucent')
            sphere.translate(t[0], t[1], t[2])
            self.viewer.addItem(sphere)
            self.camera_items.append((cam_name, sphere, t, R, cam_model))
        else:
            frustum_size = size * 5
            frustum_vertices = np.array([
                [0, 0, 0],
                [1,  1, -2],
                [1, -1, -2],
                [-1,  1, -2],
                [-1, -1, -2],
            ]) * frustum_size

            frustum_vertices = frustum_vertices @ - R + t

            lines = [
                (frustum_vertices[0], frustum_vertices[1]),
                (frustum_vertices[0], frustum_vertices[2]),
                (frustum_vertices[0], frustum_vertices[3]),
                (frustum_vertices[0], frustum_vertices[4]),
                (frustum_vertices[1], frustum_vertices[2]),
                (frustum_vertices[2], frustum_vertices[4]),
                (frustum_vertices[4], frustum_vertices[3]),
                (frustum_vertices[3], frustum_vertices[1])
            ]

            frustum_edges = []
            for start, end in lines:
                line = gl.GLLinePlotItem(pos=np.array([start, end]), color=(1, 1, 1, 0.5), width=1, antialias=True)
                self.viewer.addItem(line)
                frustum_edges.append(line)

            self.camera_items.append((cam_name, frustum_edges, t, R, cam_model))

    def show_placeholder(self):
        """球体上に「No reconstruction」という文字を表示"""
        # 球体の基本設定
        radius = 10
        rows, cols = 20, 40

        # 球体のメッシュを生成
        sphere_mesh = gl.MeshData.sphere(rows=rows, cols=cols)
        sphere = gl.GLMeshItem(meshdata=sphere_mesh, color=(0.3, 0.3, 0.3, 0.5), smooth=True, drawEdges=True)
        sphere.scale(radius, radius, radius)
        self.viewer.addItem(sphere)

    def move_to_camera(self, image_name):
        """Move viewpoint to the position associated with the specified image."""
        for name, _, position, rotation_matrix, _ in self.camera_items:
            if name == image_name:
                logger.info(f"Moving to camera position for image '{image_name}'")
                self.viewer.setCameraPosition(pos=QVector3D(position[0], position[1], position[2]), distance=20)

                # Calculate pitch and yaw from the rotation matrix
                yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                pitch = np.arcsin(-rotation_matrix[2, 0])

                # Set viewpoint rotation
                self.viewer.setCameraPosition(elevation=np.degrees(pitch), azimuth=np.degrees(yaw))
                break

    def highlight_camera(self, cam_name, highlight_color=(1, 0, 0, 0.8)):
        """Highlight the specified camera."""
        for name, item, _, _, cam_model in self.camera_items:
            if name == cam_name:
                logger.info(f"Highlighting camera: {cam_name}")
                if cam_model in ["spherical", "equirectangular"]:
                    item.setColor(highlight_color)
                else:
                    for edge in item:
                        edge.setData(color=highlight_color)
            else:
                # Reset to default color
                if cam_model in ["spherical", "equirectangular"]:
                    item.setColor((1, 1, 1, 0.3))
                else:
                    for edge in item:
                        edge.setData(color=(1, 1, 1, 0.5))

    def on_camera_image_tree_click(self, image_name):
        """クリックされた画像に関連付けられたカメラをハイライト"""
        self.highlight_camera(image_name)

    def on_camera_image_tree_double_click(self, image_name):
        """ダブルクリックされた画像に関連付けられたカメラ位置に移動"""
        self.move_to_camera(image_name)