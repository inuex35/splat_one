# app/main_app.py
import os
from PyQt5.QtWidgets import QMainWindow, QToolBar, QAction, QFileDialog, QWidget, QVBoxLayout
from app.mask_manager import MaskManager
from app.point_cloud_visualizer import MainWindow, PointCloudVisualizer

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM2 Application")
        self.setGeometry(100, 100, 1200, 800)

        # Toolbar setup
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Load Workdir action
        load_workdir_action = QAction("Load Workdir", self)
        load_workdir_action.triggered.connect(self.load_workdir)
        toolbar.addAction(load_workdir_action)

        # Initialize MaskManager and Visualizer
        self.mask_manager = None
        self.visualizer = MainWindow("reconstruction_example.json")

        # Layout for main content
        layout = QVBoxLayout()
        layout.addWidget(self.visualizer)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_workdir(self):
        """作業ディレクトリのロード"""
        workdir = QFileDialog.getExistingDirectory(self, "Select Workdir")
        if not workdir:
            return
        
        img_dir = os.path.join(workdir, "images")
        mask_dir = os.path.join(workdir, "masks")
        os.makedirs(mask_dir, exist_ok=True)

        # MaskManager の初期化
        checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
        config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.mask_manager = MaskManager(checkpoint_path, config_path, mask_dir)
        
        # サンプル点群データのロードと表示
        points = np.random.rand(100, 3) * 100  # ランダムな点群データ
        colors = np.random.rand(100, 4)        # ランダムな色
        self.visualizer.load_and_display_point_cloud(points, colors)
