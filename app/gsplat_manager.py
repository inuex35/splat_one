import os
import json
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from scipy.spatial.transform import Rotation  # 回転変換用
from utils.logger import setup_logger
import nerfview
from utils.gsplat_utils.gsplat_trainer import Runner, Config
from typing_extensions import assert_never

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
        work_dir: 作業ディレクトリ。ここから画像位置情報のJSONファイルを読み込む。
        """
        super().__init__(parent)
        self.workdir = work_dir
        # 画像位置情報ファイルのパス（例: image_poses.json）
        self.reconstruction_json_path = os.path.join(self.workdir, "reconstruction.json")
        # Runnerの初期化（Viewerは不要なのでdisable_viewer=True）
        cfg = Config(
            data_dir=self.workdir,
            result_dir=os.path.join(self.workdir, "results"),
            disable_viewer=True,
            max_steps=1
        )
        self.runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
        
        # カメラ情報を保持するリスト。各要素は (name, c2w) 形式。
        self.load_camera_data()

        # レンダリング結果表示用の QLabel
        self.image_label = QLabel("Rendering result")
        self.image_label.setAlignment(Qt.AlignCenter)
        
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

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

    def on_camera_item_clicked(self, item, column):
        """ツリービューのカメラがクリックされたとき、そのカメラ位置でレンダリングを実行する"""
        shot_name = item.text(0)
        # カメラ情報を検索
        for cam in self.cameras:
            if cam[0] == shot_name:
                _, rotation, position, cam_type = cam
                # ここでは c2w 行列を、回転行列と平行移動から構築する例
                c2w = np.eye(4)
                c2w[:3, :3] = rotation  # 回転部分
                c2w[:3, 3] = position    # 位置情報
                #camera_param = self.runner.get_camera_param(cam_type)
                # JSONから取得したカメラ情報を用いてカメラ状態を構築
                camera_state = nerfview.CameraState(
                fov=90,
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

    def on_camera_image_tree_click(self, image_name):
        """シングルクリックされた画像に関連付けられたカメラをハイライト"""
        self.highlight_camera(image_name)

    def on_camera_image_tree_double_click(self, image_name):
        """ダブルクリックされた画像に関連付けられたカメラ位置に移動"""
        self.move_to_camera(image_name)

    def highlight_camera(self, cam_name, highlight_color=(1, 0, 0, 0.8)):
        """指定されたカメラをハイライトする処理（例）"""
        for name, item, position, rotation_matrix, cam_model in self.cameras:
            if name == cam_name:
                logger.info(f"Highlighting camera: {cam_name}")
                if cam_model in ["spherical", "equirectangular"]:
                    item.setColor(highlight_color)
                else:
                    for edge in item:
                        edge.setData(color=highlight_color)
            else:
                if cam_model in ["spherical", "equirectangular"]:
                    item.setColor((1, 1, 1, 0.3))
                else:
                    for edge in item:
                        edge.setData(color=(1, 1, 1, 0.5))

    def move_to_camera(self, image_name):
        """
        指定された画像名に対応するカメラ位置へビューを移動する処理の例です。
        ここでは、保持している self.cameras（各要素：(name, rotation, position, cam_type)）
        から該当するものを探し、c2w行列を再構築して runner のレンダリング関数を呼び出すことで、
        新たなレンダリング結果（ビュー）に更新します。
        """
        for name, rotation, position, _, _ in self.cameras:
            if name == image_name:
                logger.info(f"Moving to camera position for image '{image_name}'")
                # c2w行列を作成（回転と平行移動を反映）
                c2w = np.eye(4)
                c2w[:3, :3] = rotation
                c2w[:3, 3] = position
                #camera_param = self.runner.get_camera_param(cam_type)
                # JSONから取得したカメラ情報を用いてカメラ状態を構築
                camera_state = nerfview.CameraState(
                fov=90,
                aspect=1.0,
                c2w=c2w,
                )

                # オプション：回転から yaw, pitch を計算（ビューア操作の参考用）
                yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
                pitch = np.arcsin(-rotation[2, 0])
                logger.info(f"Position: {position}, Yaw: {np.degrees(yaw):.1f}, Pitch: {np.degrees(pitch):.1f}")

                # ここで、例えば runner._viewer_render_fn() を呼び出して新しいレンダリング画像を取得
                img_wh = (640, 480)
                render = self.runner._viewer_render_fn(camera_state, img_wh)
                render_uint8 = (np.clip(render, 0, 1) * 255).astype(np.uint8)
                height, width, channels = render_uint8.shape
                bytes_per_line = channels * width
                qimage = QImage(render_uint8.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                self.image_label.setPixmap(pixmap)
                break


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
