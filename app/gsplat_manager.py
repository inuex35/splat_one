import os
import json
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QDoubleSpinBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from scipy.spatial.transform import Rotation  # For rotation conversion
from utils.logger import setup_logger
import nerfview
from utils.gsplat_utils.gsplat_trainer import Runner, Config
from utils.datasets.opensfm import *
from typing_extensions import assert_never
from pyproj import Proj
import threading

logger = setup_logger()

def load_reconstruction(file_path: str):
    """Load reconstruction data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Return list under "reconstructions" key if exists
        if "reconstructions" in data:
            return data["reconstructions"]
        return data
    except FileNotFoundError:
        logger.error(f"File {file_path} not found.")
        return None
    except json.JSONDecodeError:
        logger.error(f"File {file_path} is not valid JSON.")
        return None

class GsplatManager(QWidget):
    def __init__(self, work_dir, parent=None):
        """
        work_dir: Working directory. Reconstruction JSON is loaded from here.
        """
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)  # Enable key events in parent widget
        self.workdir = work_dir
        self.reconstruction_json_path = os.path.join(self.workdir, "reconstruction.json")
        cfg = Config(
            data_dir=self.workdir,
            result_dir=os.path.join(self.workdir, "results"),
            disable_viewer=True,
            max_steps=30000
        )
        self.runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
        self.load_camera_data()
        self.selected_cam_model = None

        # Set default configurable parameters
        self.translation_delta = 0.1
        self.rotation_delta = np.deg2rad(5)  # in radians

        # Create QLabel for rendering result and disable its focus
        self.image_label = QLabel("Rendering result")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFocusPolicy(Qt.NoFocus)

        # Create overlay buttons (camera mode and auto update)
        self.create_overlay_buttons()

        # Create training button and disable its focus
        self.start_training_button = QPushButton("Start Training")
        self.start_training_button.clicked.connect(self.start_training)
        self.start_training_button.setFocusPolicy(Qt.NoFocus)

        # Create spin boxes to configure translation and rotation step
        self.translation_spin = QDoubleSpinBox(self)
        self.translation_spin.setRange(0.01, 10.0)
        self.translation_spin.setSingleStep(0.01)
        self.translation_spin.setValue(self.translation_delta)
        self.translation_spin.valueChanged.connect(self.update_translation_delta)

        self.rotation_spin = QDoubleSpinBox(self)
        self.rotation_spin.setRange(1.0, 90.0)
        self.rotation_spin.setSingleStep(1.0)
        self.rotation_spin.setValue(5.0)  # in degrees
        self.rotation_spin.valueChanged.connect(self.update_rotation_delta)

        # Arrange configuration spin boxes in a horizontal layout
        config_layout = QHBoxLayout()
        config_layout.addWidget(QLabel("Translation Step:"))
        config_layout.addWidget(self.translation_spin)
        config_layout.addWidget(QLabel("Rotation Step (deg):"))
        config_layout.addWidget(self.rotation_spin)

        # Set up main layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(config_layout)
        layout.addWidget(self.start_training_button)
        self.setLayout(layout)
        self.training_running = False

        # Initialize QTimer for auto update
        self.auto_update_timer = QTimer(self)
        self.auto_update_timer.timeout.connect(self.auto_update)

        # Ensure the parent widget has focus
        self.setFocus()

    def create_overlay_buttons(self):
        # Create a button for perspective mode as a child of image_label
        self.btn_perspective = QPushButton("Pinhole", self.image_label)
        self.btn_perspective.setStyleSheet("background-color: rgba(255, 255, 255, 150);")
        self.btn_perspective.resize(100, 30)
        self.btn_perspective.move(10, 10)  # Position at top-left
        self.btn_perspective.clicked.connect(lambda: self.set_camera_model("pinhole"))

        # Create a button for spherical mode as a child of image_label
        self.btn_spherical = QPushButton("Spherical", self.image_label)
        self.btn_spherical.setStyleSheet("background-color: rgba(255, 255, 255, 150);")
        self.btn_spherical.resize(100, 30)
        self.btn_spherical.move(120, 10)  # Position to the right of perspective
        self.btn_spherical.clicked.connect(lambda: self.set_camera_model("spherical"))
        
        # Create a button for auto update as a child of image_label
        self.btn_auto_update = QPushButton("Auto Update: OFF", self.image_label)
        self.btn_auto_update.setStyleSheet("background-color: rgba(255, 255, 255, 150);")
        self.btn_auto_update.resize(130, 30)
        self.btn_auto_update.move(230, 10)  # Position to the right of spherical
        self.btn_auto_update.setCheckable(True)
        self.btn_auto_update.clicked.connect(self.toggle_auto_update)

    def update_translation_delta(self, value):
        # Update the translation delta from spin box value
        self.translation_delta = value
        logger.info(f"Translation delta updated: {value}")

    def update_rotation_delta(self, value):
        # Update the rotation delta (convert degrees to radians)
        self.rotation_delta = np.deg2rad(value)
        logger.info(f"Rotation delta updated: {value} degrees")

    def set_camera_model(self, model: str):
        # Update the selected camera model and update button styles for visual feedback
        self.selected_cam_model = model
        if model == "pinhole":
            self.btn_perspective.setStyleSheet("background-color: rgba(0, 200, 0, 150);")
            self.btn_spherical.setStyleSheet("background-color: rgba(255, 255, 255, 150);")
            self.move_to_camera(self.selected_image_name)
        else:
            self.btn_spherical.setStyleSheet("background-color: rgba(0, 200, 0, 150);")
            self.btn_perspective.setStyleSheet("background-color: rgba(255, 255, 255, 150);")
            self.move_to_camera(self.selected_image_name)
        logger.info(f"Selected camera model: {model}")

    def toggle_auto_update(self):
        # Toggle the auto update timer based on button state
        if self.btn_auto_update.isChecked():
            self.btn_auto_update.setText("Auto Update: ON")
            self.auto_update_timer.start(1000)  # Update every 1000 ms
            logger.info("Auto update enabled.")
        else:
            self.btn_auto_update.setText("Auto Update: OFF")
            self.auto_update_timer.stop()
            logger.info("Auto update disabled.")

    def auto_update(self):
        # Called by QTimer; update the current image rendering if an image is selected
        if hasattr(self, "selected_image_name") and self.selected_image_name:
            self.move_to_camera(self.selected_image_name)

    def start_training(self):
        """Toggle: Start training if not running, else signal to stop."""
        if not self.training_running:
            # Start training
            self.training_running = True
            self.start_training_button.setText("Stop Training")
            self.runner.stop_training = False  # Reset stop flag
            self.train_thread = threading.Thread(target=self.runner.train)
            self.train_thread.daemon = True
            self.train_thread.start()
            logger.info("Training started in background.")
        else:
            # Stop training
            self.runner.stop_training = True
            self.training_running = False
            self.start_training_button.setText("Start Training")
            logger.info("Training stop requested.")

    def load_camera_data(self):
        """Load reconstruction data from JSON and extract point cloud and camera information."""
        data = load_reconstruction(self.reconstruction_json_path)
        if data is None:
            logger.error("再構築データが読み込めませんでした。")
            return

        # Use only the first reconstruction as an example
        reconstruction = data[0]
        if "points" not in reconstruction or "shots" not in reconstruction:
            logger.warning("Data format is incorrect; missing 'points' or 'shots'")
            return

        self.cameras = []  # Initialize list
        for shot_name, shot in reconstruction["shots"].items():
            try:
                # Convert axis-angle rotation to rotation matrix
                axis_angle = np.array(shot.get("rotation", [0, 0, 0]), dtype=np.float64)
                rotation = Rotation.from_rotvec(axis_angle).as_matrix()

                # Get translation
                translation = np.array(shot.get("translation", [0, 0, 0]), dtype=np.float64)

                # Compute camera position using inverse transformation of rotation.T on translation
                position = -rotation.T @ translation

                # Camera type
                cam_type = reconstruction["cameras"][shot["camera"]]["projection_type"]
                cam_name = shot["camera"]
                self.cameras.append((shot_name, rotation, position, cam_type, cam_name))
                
            except Exception as e:
                logger.error(f"Error processing shot '{shot_name}': {e}")
                continue

    def read_opensfm(self, reconstructions):
        """Extracts camera and image information from OpenSfM reconstructions."""
        self.images = {}
        i = 0
        reference_lat_0 = reconstructions[0]["reference_lla"]["latitude"]
        reference_lon_0 = reconstructions[0]["reference_lla"]["longitude"]
        reference_alt_0 = reconstructions[0]["reference_lla"]["altitude"]
        e2u_zone = int(divmod(reference_lon_0, 6)[0]) + 31
        e2u_conv = Proj(proj='utm', zone=e2u_zone, ellps='WGS84')
        reference_x_0, reference_y_0 = e2u_conv(reference_lon_0, reference_lat_0)
        if reference_lat_0 < 0:
            reference_y_0 += 10000000
        
        self.cameras = {}
        self.camera_names = {}
        cam_id = 1
        
        for reconstruction in reconstructions:
            # Parse cameras.
            for i, camera in enumerate(reconstruction["cameras"]):
                camera_name = camera
                camera_info = reconstruction["cameras"][camera]
                if camera_info['projection_type'] in ['spherical', 'equirectangular']:
                    camera_id = 0
                    model = "SPHERICAL"
                    width = reconstruction["cameras"][camera]["width"]
                    height = reconstruction["cameras"][camera]["height"]
                    params = np.array([0])
                    self.cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params, panorama=True)
                    self.camera_names[camera_name] = camera_id
                elif reconstruction["cameras"][camera]['projection_type'] == "perspective":
                    model = "SIMPLE_PINHOLE"
                    width = reconstruction["cameras"][camera]["width"]
                    height = reconstruction["cameras"][camera]["height"]
                    f = reconstruction["cameras"][camera]["focal"] * width
                    k1 = reconstruction["cameras"][camera]["k1"]
                    k2 = reconstruction["cameras"][camera]["k2"]
                    params = np.array([f, width / 2, height / 2, k1, k2])
                    camera_id = cam_id
                    self.cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params, panorama=False)
                    self.camera_names[camera_name] = camera_id
                    cam_id += 1
            
            reference_lat = reconstruction["reference_lla"]["latitude"]
            reference_lon = reconstruction["reference_lla"]["longitude"]
            reference_alt = reconstruction["reference_lla"]["altitude"]
            reference_x, reference_y = e2u_conv(reference_lon, reference_lat)
            if reference_lat < 0:
                reference_y += 10000000        
            for j, shot in enumerate(reconstruction["shots"]):
                translation = reconstruction["shots"][shot]["translation"]
                rotation = reconstruction["shots"][shot]["rotation"]
                qvec = angle_axis_to_quaternion(rotation)
                diff_ref_x = reference_x - reference_x_0
                diff_ref_y = reference_y - reference_y_0
                diff_ref_alt = reference_alt - reference_alt_0
                tvec = np.array([translation[0], translation[1], translation[2]])
                diff_ref = np.array([diff_ref_x, diff_ref_y, diff_ref_alt])
                camera_name = reconstruction["shots"][shot]["camera"]
                camera_id = self.camera_names.get(camera_name, 0)
                image_id = j
                image_name = shot
                xys = np.array([0, 0])
                point3D_ids = np.array([0, 0])
                self.images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name, xys=xys, point3D_ids=point3D_ids, diff_ref=diff_ref)

    def on_camera_item_clicked(self, item, column):
        """When a camera in the tree view is clicked, render from its position."""
        shot_name = item.text(0)
        self.move_to_camera(shot_name)

    def on_camera_image_tree_click(self, image_name):
        """Highlight the camera corresponding to the clicked image."""
        self.move_to_camera(image_name)

    def on_camera_image_tree_double_click(self, image_name):
        """Move to the camera position associated with the double-clicked image."""
        self.move_to_camera(image_name)

    def move_to_camera(self, image_name):
        """
        Move the view to the camera position corresponding to the specified image.
        Retrieves the data directly from the dictionary to reduce overhead.
        """
        self.selected_image_name = image_name
        import time
        start_total = time.time()  # Start overall timer

        # Retrieve the sample directly from the dataset using image name
        data = self.runner.allset.get_data_by_image_name(image_name)
        if data is None:
            logger.error(f"Image '{image_name}' not found.")
            return

        logger.info(f"Moving to camera position for image '{image_name}'")

        # Measure transfer time
        transfer_start = time.time()
        c2w = data["camtoworld"].to(self.runner.device)
        transfer_time = time.time() - transfer_start

        # Construct the CameraState
        self.camera_state = nerfview.CameraState(
            fov=90,
            aspect=1.0,
            c2w=c2w,
        )

        # Get image size from the data tensor shape
        img_tensor = data["image"]
        h, w, _ = img_tensor.shape
        img_wh = (w, h)  # (width, height)
        if self.selected_cam_model == "pinhole":
            h = w

        # Measure rendering time
        render_start = time.time()
        render = self.runner._viewer_render_fn(self.camera_state, img_wh, camera_model=self.selected_cam_model)
        render_time = time.time() - render_start

        # Measure post-processing time and scale the image to fit the label
        post_start = time.time()
        render_uint8 = (np.clip(render, 0, 1) * 255).astype(np.uint8)
        height, width, channels = render_uint8.shape
        bytes_per_line = channels * width
        qimage = QImage(render_uint8.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        # Scale pixmap to fit the label's current size while keeping aspect ratio
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        post_time = time.time() - post_start

        total_time = time.time() - start_total

        # Log the timings
        logger.info(
            f"Timing: Transfer={transfer_time:.4f}s, Render={render_time:.4f}s, "
            f"PostProcessing={post_time:.4f}s, Total={total_time:.4f}s"
        )

    def keyPressEvent(self, event):
        """
        Override keyPressEvent to update camera position and orientation.
        W/A/S/D: Move forward/left/back/right.
        Q/E: Move up/down.
        Arrow keys: Rotate (yaw and pitch).
        """
        if not hasattr(self, "camera_state"):
            return

        # Convert current c2w to numpy
        c2w = self.camera_state.c2w.cpu().numpy() if hasattr(self.camera_state.c2w, 'cpu') else self.camera_state.c2w

        R = c2w[:3, :3]
        t = c2w[:3, 3]

        forward = -R[:, 2]  # Forward direction (-Z)
        right = R[:, 0]     # Right direction (X)
        up = R[:, 1]        # Up direction (Y)

        key = event.key()
        if key == Qt.Key_W:
            t = t - self.translation_delta * forward
        elif key == Qt.Key_S:
            t = t + self.translation_delta * forward
        elif key == Qt.Key_A:
            t = t - self.translation_delta * right
        elif key == Qt.Key_D:
            t = t + self.translation_delta * right
        elif key == Qt.Key_Q:
            t = t + self.translation_delta * up
        elif key == Qt.Key_E:
            t = t - self.translation_delta * up
        elif key == Qt.Key_Left:
            R_delta = Rotation.from_rotvec(-self.rotation_delta * up).as_matrix()
            R = R_delta @ R
        elif key == Qt.Key_Right:
            R_delta = Rotation.from_rotvec(self.rotation_delta * up).as_matrix()
            R = R_delta @ R
        elif key == Qt.Key_Up:
            R_delta = Rotation.from_rotvec(self.rotation_delta * right).as_matrix()
            R = R_delta @ R
        elif key == Qt.Key_Down:
            R_delta = Rotation.from_rotvec(-self.rotation_delta * right).as_matrix()
            R = R_delta @ R
        else:
            return

        new_c2w = np.eye(4)
        new_c2w[:3, :3] = R
        new_c2w[:3, 3] = t
        self.camera_state.c2w = torch.from_numpy(new_c2w).to(self.runner.device)

        # Re-render the scene with updated camera state
        data = self.runner.allset.get_data_by_image_name(self.selected_image_name)
        if data is None:
            logger.error(f"Image '{self.selected_image_name}' not found for key update.")
            return
        img_tensor = data["image"]
        h, w, _ = img_tensor.shape
        img_wh = (w, h)
        render = self.runner._viewer_render_fn(self.camera_state, img_wh, camera_model=self.selected_cam_model)
        render_uint8 = (np.clip(render, 0, 1) * 255).astype(np.uint8)
        height, width, channels = render_uint8.shape
        bytes_per_line = channels * width
        qimage = QImage(render_uint8.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)

    # Example working directory
    workdir = "./workdir"
    # Initialize Runner (viewer disabled)
    cfg = Config(
        data_dir=workdir,
        result_dir=os.path.join(workdir, "results"),
        disable_viewer=True,
        max_steps=1
    )
    runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
    
    # Example JSON file path for image poses (e.g., image_poses.json)
    pose_json_path = os.path.join(workdir, "image_poses.json")
    
    widget = GsplatManager(workdir)
    widget.show()
    sys.exit(app.exec_())
