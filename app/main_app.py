# main_app.py
import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QToolBar, QAction, QFileDialog, QMessageBox,
    QDialog, QVBoxLayout, QPushButton, QLabel, QWidget, QTabWidget, QSplitter,
    QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem, QTextEdit, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt, QTimer
from opensfm.dataset import DataSet
from app.mask_manager import MaskManager  # Import the updated MaskManager class
from app.feature_extractor import FeatureExtractor  # 新しく追加
from app.point_cloud_visualizer import PointCloudVisualizer  # Import the PointCloudVisualizer class
class StartDialog(QDialog):
    """Dialog to offer options at startup."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select an Option")
        self.setFixedSize(300, 150)
        layout = QVBoxLayout()

        self.label = QLabel("Please select an option:")
        layout.addWidget(self.label, alignment=Qt.AlignCenter)

        self.new_button = QPushButton("New Reconstruction")
        self.new_button.clicked.connect(self.accept_new)
        layout.addWidget(self.new_button)

        self.existing_button = QPushButton("Start from Existing Data")
        self.existing_button.clicked.connect(self.accept_existing)
        layout.addWidget(self.existing_button)

        self.selection = None
        self.setLayout(layout)

    def accept_new(self):
        self.selection = "new"
        self.accept()

    def accept_existing(self):
        self.selection = "existing"
        self.accept()

class MaskManagerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mask_manager = parent.mask_manager  # Reference to the existing MaskManager
        layout = QVBoxLayout()
        layout.addWidget(self.mask_manager)
        self.setLayout(layout)

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM2 Mask Creator")
        self.setGeometry(100, 100, 1200, 800)
        self.image_list = []  # Initialize image_list

        # Set up toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Initialize placeholders
        self.mask_manager = None
        self.mask_manager_widget = None  # Add this line
        self.workdir = None
        self.masks_tab_initialized = False  # Flag to check if Masks tab is initialized
        self.features_tab_initialized = False
        self.feature_extractor = None  # FeatureExtractor インスタンスを格納
        self.reconstruct_tab_initialized = False

        # Set up tab widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Initialize tabs
        self.init_tabs()

        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # After showing the main window, display the start dialog
        QTimer.singleShot(0, self.show_start_dialog)

    def init_tabs(self):
        """Initialize all tabs."""
        self.images_tab = QWidget()
        self.masks_tab = QWidget()
        self.features_tab = QWidget()
        self.matching_tab = QWidget()
        self.reconstruct_tab = QWidget()
        self.gsplat_tab = QWidget()

        # Add tabs to the tab widget
        self.tab_widget.addTab(self.images_tab, "Images")
        self.tab_widget.addTab(self.masks_tab, "Masks")
        self.tab_widget.addTab(self.features_tab, "Features")
        self.tab_widget.addTab(self.matching_tab, "Matching")
        self.tab_widget.addTab(self.reconstruct_tab, "Reconstruct")
        self.tab_widget.addTab(self.gsplat_tab, "Gsplat")

        # Initialize each tab's content
        self.init_images_tab()

    def init_images_tab(self):
        """Initialize the Images tab."""
        layout = QSplitter(Qt.Horizontal)

        # Left side: Tree of images grouped by camera
        self.camera_image_tree = QTreeWidget()
        self.camera_image_tree.setHeaderLabel("Cameras and Images")
        self.camera_image_tree.setFixedWidth(250)  # 固定幅を設定
        layout.addWidget(self.camera_image_tree)

        # Right side: Splitter with image viewer and EXIF display
        right_splitter = QSplitter(Qt.Vertical)

        # Image viewer
        self.image_viewer = QLabel("Image Viewer")
        self.image_viewer.setAlignment(Qt.AlignCenter)
        self.image_viewer.setMinimumHeight(300)
        right_splitter.addWidget(self.image_viewer)

        # EXIF data display using QTableWidget
        self.exif_table = QTableWidget()
        self.exif_table.setColumnCount(2)
        self.exif_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.exif_table.horizontalHeader().setStretchLastSection(True)
        right_splitter.addWidget(self.exif_table)

        layout.addWidget(right_splitter)

        # ストレッチファクターの設定
        layout.setStretchFactor(0, 1)  # 左側を1
        layout.setStretchFactor(1, 4)  # 右側を4

        self.images_tab.setLayout(QVBoxLayout())
        self.images_tab.layout().addWidget(layout)

        # Connect signals
        self.camera_image_tree.itemClicked.connect(self.display_image_and_exif)

    def init_masks_tab(self):
        """Initialize the Masks tab."""
        layout = QSplitter(Qt.Horizontal)

        # Left side: Tree of images grouped by camera (like other tabs)
        self.camera_image_tree = QTreeWidget()
        self.camera_image_tree.setHeaderLabel("Cameras and Images")
        self.camera_image_tree.setFixedWidth(250)  # 固定幅を設定
        layout.addWidget(self.camera_image_tree)

        if self.workdir is None:
            QMessageBox.warning(self, "Error", "Work directory is not set.")
            return

        if self.mask_manager is None:
            # Initialize MaskManager
            checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
            config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
            mask_dir = os.path.join(self.workdir, "masks")
            img_dir = os.path.join(self.workdir, "images")
            if not self.image_list:
                # Get image list
                img_dir = os.path.join(self.workdir, "images")
                self.image_list = [
                    f for f in os.listdir(img_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
            self.mask_manager = MaskManager(
                checkpoint_path, config_path, mask_dir, img_dir, self.image_list
            )

        # Right side: MaskManager widget
        self.mask_manager_widget = MaskManagerWidget(parent=self)
        layout.addWidget(self.mask_manager_widget)

        # Set stretch factors
        layout.setStretchFactor(0, 1)  # Left side (image tree)
        layout.setStretchFactor(1, 4)  # Right side (MaskManager)

        # Set layout for mask tab
        self.masks_tab.setLayout(QVBoxLayout())
        self.masks_tab.layout().addWidget(layout)

        # Populate the camera image tree
        self.populate_tree_with_camera_data(self.camera_image_tree)

        # Connect signals
        self.camera_image_tree.itemClicked.connect(self.display_mask)

    def init_features_tab(self):
        """Initialize the Features tab."""
        layout = QSplitter(Qt.Horizontal)

        # Left side: Tree of images grouped by camera
        self.camera_image_tree = QTreeWidget()
        self.camera_image_tree.setHeaderLabel("Cameras and Images")
        self.camera_image_tree.setFixedWidth(250)  # 固定幅を設定
        layout.addWidget(self.camera_image_tree)

        # Right side: FeatureExtractor widget
        self.feature_extractor = FeatureExtractor(self.workdir, self.image_list)
        layout.addWidget(self.feature_extractor)

        # Set layout for features tab
        self.features_tab.setLayout(QVBoxLayout())
        self.features_tab.layout().addWidget(layout)

        self.populate_tree_with_camera_data(self.camera_image_tree)

        # Connect signals
        self.camera_image_tree.itemClicked.connect(self.display_features_for_image)


    def init_reconstruct_tab(self):
        """Initialize the Reconstruct tab."""
        layout = QSplitter(Qt.Horizontal)

        # Left side: Tree of images grouped by camera
        self.camera_image_tree = QTreeWidget()
        self.camera_image_tree.setHeaderLabel("Cameras and Images")
        self.camera_image_tree.setFixedWidth(250)  # 固定幅を設定
        layout.addWidget(self.camera_image_tree)

        # Right side: PointCloudVisualizer widget
        self.pointcloud_viewer = PointCloudVisualizer(file_path=self.workdir + "/reconstruction.json")
        layout.addWidget(self.pointcloud_viewer)

        # Set layout for reconstruct tab
        self.reconstruct_tab.setLayout(QVBoxLayout())
        self.reconstruct_tab.layout().addWidget(layout)

        # Connect signals
        self.camera_image_tree.itemClicked.connect(self.handle_camera_image_tree_click)
        self.camera_image_tree.itemDoubleClicked.connect(self.handle_camera_image_tree_double_click)

        # Populate the tree with camera data
        self.populate_tree_with_camera_data(self.camera_image_tree)

    def on_tab_changed(self, index):
        """Handle actions when a tab is changed."""
        tab_name = self.tab_widget.tabText(index)
        
        if tab_name == "Masks":
            if not self.masks_tab_initialized:
                self.init_masks_tab()
                self.masks_tab_initialized = True
            elif self.mask_manager.sam_model is None:
                self.mask_manager.init_sam_model()
        else:
            if self.mask_manager is not None:
                self.mask_manager.unload_sam_model()

        if tab_name == "Reconstruct":
            if not self.reconstruct_tab_initialized:
                self.init_reconstruct_tab()
                self.reconstruct_tab_initialized = True
            elif self.pointcloud_viewer:
                self.pointcloud_viewer.update_visualization()

        elif tab_name == "Features":
            # Featureタブの初期化処理
            if not self.features_tab_initialized:
                self.init_features_tab()  # Featuresタブの初期化メソッドを呼び出す
                self.features_tab_initialized = True

        

    def show_start_dialog(self):
        """Display a dialog to select an option at startup."""
        dialog = StartDialog(self)
        dialog.setWindowModality(Qt.ApplicationModal)
        if dialog.exec_() == QDialog.Accepted:
            if dialog.selection == "new":
                self.start_new_reconstruction()
            elif dialog.selection == "existing":
                self.start_from_existing_data()

    def start_new_reconstruction(self):
        """Process to start a new reconstruction."""
        self.load_workdir(new_project=True)

    def start_from_existing_data(self):
        """Process to start from existing data."""
        self.load_workdir(new_project=False)

    def load_workdir(self, new_project):
        """Load the work directory and initialize MaskManager."""
        self.workdir = QFileDialog.getExistingDirectory(self, "Select Workdir")
        if not self.workdir:
            return

        # images folder check
        img_dir = os.path.join(self.workdir, "images")
        exif_dir = os.path.join(self.workdir, "exif")
        if not os.path.exists(img_dir):
            QMessageBox.warning(self, "Error", "images folder does not exist.")
            return
        else:
            try:
                dataset = DataSet(self.workdir)
                from opensfm.actions import extract_metadata
                extract_metadata.run_dataset(dataset)
                QMessageBox.information(self, "Success", "Metadata extracted successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to extract metadata: {e}")
                return

            # Create or check masks folder
            mask_dir = os.path.join(self.workdir, "masks")
            if new_project:
                os.makedirs(mask_dir, exist_ok=True)
            else:
                if not os.path.exists(mask_dir):
                    os.makedirs(mask_dir, exist_ok=True)

            # Set self.image_list here
            self.image_list = [
                f for f in os.listdir(img_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

            if self.image_list:
                # 再表示: imagesタブ用のimages_treeにデータを再設定
                self.populate_tree_with_camera_data(self.camera_image_tree)
                QMessageBox.information(self, "Success", "Workdir loaded successfully.")
            else:
                QMessageBox.warning(self, "Error", "No images found in the images folder.")
                return


    def populate_tree_with_camera_data(self, tree_widget):
        """Populate the specified tree widget with camera and image data."""
        tree_widget.clear()
        camera_groups = {}
        exif_dir = os.path.join(self.workdir, "exif")

        for image_name in self.image_list:
            exif_file = os.path.join(exif_dir, image_name + '.exif')
            if os.path.exists(exif_file):
                with open(exif_file, 'r') as f:
                    exif_data = json.load(f)
                camera = exif_data.get('camera', 'Unknown Camera')
            else:
                camera = 'Unknown Camera'

            if camera not in camera_groups:
                camera_groups[camera] = []

            camera_groups[camera].append(image_name)

        for camera, images in camera_groups.items():
            camera_item = QTreeWidgetItem(tree_widget)
            camera_item.setText(0, camera)
            for img in images:
                img_item = QTreeWidgetItem(camera_item)
                img_item.setText(0, img)

    def display_image_and_exif(self, item, column):
        """Display the selected image and its EXIF data."""
        if item.childCount() == 0 and item.parent() is not None:
            image_name = item.text(0)
            image_path = os.path.join(self.workdir, "images", image_name)
            exif_path = os.path.join(self.workdir, "exif", image_name + '.exif')

            # Display image
            if os.path.exists(image_path):
                from PyQt5.QtGui import QPixmap
                pixmap = QPixmap(image_path)
                self.image_viewer.setPixmap(pixmap.scaled(
                    self.image_viewer.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.image_viewer.setText("Image not found.")

            # Display EXIF data
            if os.path.exists(exif_path):
                with open(exif_path, 'r') as f:
                    exif_data = json.load(f)
                self.display_exif_data(exif_data)
            else:
                self.exif_table.setRowCount(0)
                self.exif_table.setHorizontalHeaderLabels(["Field", "Value"])
                self.exif_table.clearContents()
                self.exif_table.setRowCount(1)
                self.exif_table.setItem(0, 0, QTableWidgetItem("Error"))
                self.exif_table.setItem(0, 1, QTableWidgetItem("EXIF data not found."))
        else:
            # Clicked on a camera group
            pass

    def display_exif_data(self, exif_data):
        """Display EXIF data in a table."""
        self.exif_table.setRowCount(0)
        self.exif_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.exif_table.clearContents()

        # Define the fields to display and their order
        fields = [
            "make",
            "model",
            "width",
            "height",
            "projection_type",
            "focal_ratio",
            "orientation",
            "capture_time",
            "gps",
            "camera"
        ]

        for i, field in enumerate(fields):
            self.exif_table.insertRow(i)
            key_item = QTableWidgetItem(field)
            value = exif_data.get(field, "N/A")
            if isinstance(value, dict):
                value = json.dumps(value)
            elif isinstance(value, float):
                value = f"{value:.2f}"
            value_item = QTableWidgetItem(str(value))
            self.exif_table.setItem(i, 0, key_item)
            self.exif_table.setItem(i, 1, value_item)

    def display_mask(self, item, column):
        """Display the mask corresponding to the selected image."""
        if item.childCount() == 0 and item.parent() is not None:
            image_name = item.text(0)
            if self.mask_manager is not None:
                self.mask_manager.load_image_by_name(image_name)
            else:
                QMessageBox.warning(self, "Error", "Mask Manager is not initialized.")
        else:
            # Clicked on a camera group
            pass

    def display_features_for_image(self, item, column):
        """Display features for the selected image."""
        # 画像名を取得
        image_name = item.text(0)  # 修正: 列を指定
        if self.feature_extractor:
            # FeatureExtractorで選択された画像をロードして特徴を抽出
            self.feature_extractor.load_image_by_name(image_name)
        else:
            QMessageBox.warning(self, "Error", "Feature extractor is not initialized.")

    def handle_camera_image_tree_click(self, item, column):
        """Handle single click event for camera_image_tree and pass image name to PointCloudVisualizer."""
        if item.childCount() == 0 and item.parent() is not None:  # 画像項目のみ反応
            image_name = item.text(0)
            self.pointcloud_viewer.on_camera_image_tree_click(image_name)

    def handle_camera_image_tree_double_click(self, item, column):
        """Handle double click event for camera_image_tree and move to the camera position."""
        if item.childCount() == 0 and item.parent() is not None:  # 画像項目のみ反応
            image_name = item.text(0)
            self.pointcloud_viewer.on_camera_image_tree_double_click(image_name)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
