import os
import json
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QLabel, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from scipy.spatial.transform import Rotation  # 回転変換用
from utils.logger import setup_logger
import nerfview
from utils.gsplat_utils.gsplat_trainer import Runner, Config
from utils.datasets.opensfm import *
from typing_extensions import assert_never
from pyproj import Proj
import threading

logger = setup_logger()

def load_reconstruction(file_path: str):
    """JSONファイルから再構築データを読み込む関数"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        # "reconstructions" キーがある場合はそのリストを返す
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
        self.workdir = work_dir
        self.reconstruction_json_path = os.path.join(self.workdir, "reconstruction.json")
        # Initialize Runner (disable viewer)
        cfg = Config(
            data_dir=self.workdir,
            result_dir=os.path.join(self.workdir, "results"),
            disable_viewer=True,
            max_steps=30000
        )
        self.runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
        
        # Load camera data from JSON
        self.load_camera_data()

        # QLabel for rendering result
        self.image_label = QLabel("Rendering result")
        self.image_label.setAlignment(Qt.AlignCenter)
        
        # Add a button to start training (or any desired action)
        self.start_training_button = QPushButton("Start Training")
        self.start_training_button.clicked.connect(self.start_training)
        
        # Set up layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.start_training_button)
        self.setLayout(layout)
        self.training_running = False

    def start_training(self):
        """Toggle: Start training if not running, else signal to stop."""
        import threading
        if not self.training_running:
            # Training を開始
            self.training_running = True
            self.start_training_button.setText("Stop Training")
            self.runner.stop_training = False  # 停止フラグをリセット
            self.train_thread = threading.Thread(target=self.runner.train)
            self.train_thread.daemon = True
            self.train_thread.start()
            logger.info("Training started in background.")
        else:
            # Training を停止
            self.runner.stop_training = True
            self.training_running = False
            self.start_training_button.setText("Start Training")
            logger.info("Training stop requested.")

    def load_camera_data(self):
        """JSONから再構築データを読み取り、ポイントクラウドとカメラ情報を抽出する"""
        data = load_reconstruction(self.reconstruction_json_path)
        if data is None:
            logger.error("再構築データが読み込めませんでした。")
            return

        # ここでは最初の再構築データのみを利用する例
        reconstruction = data[0]
        if "points" not in reconstruction or "shots" not in reconstruction:
            logger.warning("Data format is incorrect; missing 'points' or 'shots'")
            return


        self.cameras = []  # 初期化
        for shot_name, shot in reconstruction["shots"].items():
            try:

                # 軸角（axis-angle）形式の回転情報を取得し、回転行列に変換
                axis_angle = np.array(shot.get("rotation", [0, 0, 0]), dtype=np.float64)
                rotation = Rotation.from_rotvec(axis_angle).as_matrix()

                # 平行移動（translation）の取得
                translation = np.array(shot.get("translation", [0, 0, 0]), dtype=np.float64)

                # カメラ位置は、rotation.T の逆変換で translation を適用（例）
                position = -rotation.T @ translation

                # カメラの種類
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
        #return cameras, images

    def on_camera_item_clicked(self, item, column):
        """ツリービューのカメラがクリックされたとき、そのカメラ位置でレンダリングを実行する"""
        shot_name = item.text(0)
        self.move_to_camera(shot_name)
        """
        # カメラ情報を検索
        for cam in self.cameras:
            if cam[0] == shot_name:
                _, rotation, position, cam_type = cam
                # ここでは c2w 行列を、回転行列と平行移動から構築する例
                c2w = np.eye(4)
                c2w[:3, :3] = rotation  # 回転部分
                c2w[:3, 3] = position  # 位置情報
                #camera_param = self.runner.get_camera_param(cam_type)
                # JSONから取得したカメラ情報を用いてカメラ状態を構築
                camera_state = nerfview.CameraState(
                fov=90 / 180.0 * np.pi,
                aspect=1.0,
                c2w=c2w,
                )
                # 表示解像度（例：640x480）
                img_wh = (640, 480)
                # Runner の _viewer_render_fn() を呼び出してレンダリングを実行
                render = self.runner.rasterize_splats(camera_state, img_wh)
                # 取得した画像データは [0,1] の範囲と仮定
                render_uint8 = (np.clip(render, 0, 1) * 255).astype(np.uint8)
                height, width, channels = render_uint8.shape
                bytes_per_line = channels * width
                qimage = QImage(render_uint8.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                self.image_label.setPixmap(pixmap)
                logger.info(f"{shot_name} の位置でレンダリングを実行しました。")
                break
        """

    def on_camera_image_tree_click(self, image_name):
        """シングルクリックされた画像に関連付けられたカメラをハイライト"""
        self.move_to_camera(image_name)

    def on_camera_image_tree_double_click(self, image_name):
        """ダブルクリックされた画像に関連付けられたカメラ位置に移動"""
        self.move_to_camera(image_name)

    def move_to_camera(self, image_name):
        """
        指定された画像名に対応するカメラ位置へビューを移動する処理。
        辞書から直接データを取得することで、全件走査のオーバーヘッドを削減します。
        """
        import time
        start_total = time.time()  # Start overall timer

        # Retrieve the sample directly from the dictionary
        data = self.runner.allset_dict.get(image_name)
        if data is None:
            logger.error(f"Image '{image_name}' not found.")
            return

        logger.info(f"Moving to camera position for image '{image_name}'")

        # Measure transfer time
        transfer_start = time.time()
        c2w = data["camtoworld"].to(self.runner.device)
        transfer_time = time.time() - transfer_start

        # Construct the CameraState
        camera_state = nerfview.CameraState(
            fov=90,
            aspect=1.0,
            c2w=c2w,
        )

        # 自動的に画像サイズを取得（data["image"] の shape から）
        img_tensor = data["image"]
        h, w, _ = img_tensor.shape
        img_wh = (w, h)  # (width, height)

        # Measure rendering time
        render_start = time.time()
        render = self.runner._viewer_render_fn(camera_state, img_wh)
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


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)

    # 作業ディレクトリの例
    workdir = "./workdir"
    # Runner の初期化（Viewerは不要なのでdisable_viewer=True）
    cfg = Config(
        data_dir=workdir,
        result_dir=os.path.join(workdir, "results"),
        disable_viewer=True,
        max_steps=1
    )
    runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
    
    # 画像の位置情報を記述したJSONファイルのパス（例：image_poses.json）
    pose_json_path = os.path.join(workdir, "image_poses.json")
    
    widget = GsplatManager(runner, pose_json_path)
    widget.show()
    sys.exit(app.exec_())
