import numpy as np
import os
import json
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QVector3D
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from opensfm import dataset
from opensfm.actions import reconstruct, create_tracks
from opensfm.reconstruction import ReconstructionAlgorithm
from scipy.spatial.transform import Rotation
from utils.logger import setup_logger

logger = setup_logger()

def safe_load_reconstruction(file_path: str, retries=3, delay=1):
    for attempt in range(retries):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Attempt {attempt + 1}/{retries}: Error loading {file_path}: {e}")
            time.sleep(delay)
    logger.error(f"Failed to load reconstruction data from {file_path} after {retries} attempts.")
    return None

class ReconstructionThread(QThread):
    finished = pyqtSignal()
    stopped = pyqtSignal()

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self._running = True

    def run(self):
        create_tracks.run_dataset(self.dataset)
        
        if not self._running:
            self.stopped.emit()
            return
        
        reconstruct.run_dataset(self.dataset, ReconstructionAlgorithm.INCREMENTAL)

        if not self._running:
            self.stopped.emit()
            return

        self.finished.emit()

    def stop(self):
        self._running = False

class Reconstruction(QWidget):
    def __init__(self, workdir):
        super().__init__()
        self.workdir = workdir
        self.reconstruction_file = os.path.join(workdir, "reconstruction.json")
        self.dataset = dataset.DataSet(workdir)
        self.camera_size = 1.0
        self.last_mod_time = 0
        self.camera_items = []

        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        main_layout = QVBoxLayout(self)

        self.viewer = gl.GLViewWidget()
        self.viewer.setFocusPolicy(Qt.NoFocus)
        self.viewer.setCameraPosition(distance=50)
        main_layout.addWidget(self.viewer)

        button_layout = QHBoxLayout()

        self.run_button = QPushButton("Run Reconstruction")
        self.run_button.clicked.connect(self.run_reconstruction_thread)
        button_layout.addWidget(self.run_button)

        self.stop_button = QPushButton("Stop Reconstruction")
        self.stop_button.clicked.connect(self.stop_reconstruction_thread)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        self.config_button = QPushButton("Config")
        self.config_button.clicked.connect(self.configure_reconstruction)
        button_layout.addWidget(self.config_button)

        main_layout.addLayout(button_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_for_updates)
        self.timer.start(10000)

        self.update_visualization()

    def run_reconstruction_thread(self):
        reply = QMessageBox.question(
            self, 'Confirmation',
            'Run the reconstruction process?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.run_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.thread = ReconstructionThread(self.dataset)
            self.thread.finished.connect(self.on_reconstruction_finished)
            self.thread.stopped.connect(self.on_reconstruction_stopped)
            self.thread.start()

    def stop_reconstruction_thread(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            reply = QMessageBox.question(
                self, 'Confirmation',
                'Stop the reconstruction process?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.thread.stop()
                self.thread.wait()

    def on_reconstruction_finished(self):
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.update_visualization()
        QMessageBox.information(self, "Done", "Reconstruction completed successfully.")

    def on_reconstruction_stopped(self):
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        QMessageBox.information(self, "Stopped", "Reconstruction was stopped.")

    def configure_reconstruction(self):
        QMessageBox.information(self, "Config", "Configuration dialog placeholder.")

    def check_for_updates(self):
        if os.path.exists(self.reconstruction_file):
            mod_time = os.path.getmtime(self.reconstruction_file)
            if mod_time != self.last_mod_time:
                self.last_mod_time = mod_time
                data = safe_load_reconstruction(self.reconstruction_file)
                if data:
                    self.update_visualization()

    def update_visualization(self):
        self.viewer.clear()
        self.camera_items.clear()

        data = safe_load_reconstruction(self.reconstruction_file)
        if not data:
            self.show_placeholder()
            return

        for reconstruction in data:
            points = np.array([p["coordinates"] for p in reconstruction.get("points", {}).values()], dtype=float)
            colors = np.array([p["color"] for p in reconstruction.get("points", {}).values()], dtype=float) / 255.0
            if points.size:
                scatter = gl.GLScatterPlotItem(pos=points, color=colors, size=2)
                scatter.setGLOptions('translucent')
                self.viewer.addItem(scatter)

            for shot_name, shot in reconstruction.get("shots", {}).items():
                rotation = Rotation.from_rotvec(np.array(shot["rotation"])).as_matrix()
                translation = np.array(shot["translation"])
                position = -rotation.T @ translation
                cam_type = reconstruction["cameras"][shot["camera"]]["projection_type"]
                self.add_camera_visualization(shot_name, rotation, position, cam_type, self.camera_size)

    def add_camera_visualization(self, cam_name, R, t, cam_model, size=1.0):
        if cam_model in ["spherical", "equirectangular"]:
            sphere_mesh = pg.opengl.MeshData.sphere(rows=10, cols=20, radius=size)
            sphere = gl.GLMeshItem(meshdata=sphere_mesh, color=(1, 1, 1, 0.3), smooth=True)
            sphere.translate(*t)
            self.viewer.addItem(sphere)
            self.camera_items.append((cam_name, sphere, t, R, cam_model))
        else:
            frustum_size = size * 5
            vertices = np.array([
                [0, 0, 0],
                [1, 1, -2],
                [1, -1, -2],
                [-1, 1, -2],
                [-1, -1, -2],
            ]) * frustum_size
            vertices = vertices @ -R + t

            edges = [
                (0, 1), (0, 2), (0, 3), (0, 4),
                (1, 2), (2, 4), (4, 3), (3, 1)
            ]

            lines = []
            for s, e in edges:
                line = gl.GLLinePlotItem(pos=np.array([vertices[s], vertices[e]]), color=(1, 1, 1, 0.5), width=1)
                self.viewer.addItem(line)
                lines.append(line)

            self.camera_items.append((cam_name, lines, t, R, cam_model))

    def show_placeholder(self):
        sphere_mesh = pg.opengl.MeshData.sphere(rows=10, cols=20, radius=10)
        sphere = gl.GLMeshItem(meshdata=sphere_mesh, color=(0.5, 0.5, 0.5, 0.3), smooth=True)
        self.viewer.addItem(sphere)

    def move_to_camera(self, image_name):
        for name, _, pos, R, _ in self.camera_items:
            if name == image_name:
                elevation = np.degrees(np.arcsin(-R[2, 0]))
                azimuth = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                self.viewer.setCameraPosition(pos=QVector3D(*pos), distance=20, elevation=elevation, azimuth=azimuth)
                break

    def highlight_camera(self, cam_name):
        for name, item, _, _, model in self.camera_items:
            color = (1, 0, 0, 0.8) if name == cam_name else (1, 1, 1, 0.3)
            if model in ["spherical", "equirectangular"]:
                item.setColor(color)
            else:
                for line in item:
                    line.setData(color=color)

    def on_camera_image_tree_click(self, image_name):
        self.highlight_camera(image_name)

    def on_camera_image_tree_double_click(self, image_name):
        self.move_to_camera(image_name)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Plus:
            self.camera_size *= 1.1
            self.update_visualization()
        elif event.key() == Qt.Key_Minus:
            self.camera_size /= 1.1
            self.update_visualization()
        else:
            super().keyPressEvent(event)
