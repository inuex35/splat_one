# main_app.py
import sys
import os
import json
import shutil
import importlib.util
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QToolBar, QAction, QFileDialog, QMessageBox,
    QDialog, QVBoxLayout, QPushButton, QLabel, QWidget, QTabWidget, QSplitter,
    QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem, QTextEdit, QTableWidget, QTableWidgetItem, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from opensfm.dataset import DataSet
from app.mask_manager import MaskManager  # Import the updated MaskManager class
from app.feature_extractor import FeatureExtractor  # 新しく追加
from app.feature_matching import FeatureMatching  # 新しく追加
from app.point_cloud_visualizer import Reconstruction  # Import the PointCloudVisualizer class

class CameraModelEditor(QDialog):
    """カメラモデル編集ダイアログ"""
    def __init__(self, camera_models, workdir, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Camera Models")
        self.setFixedSize(600, 400)

        self.camera_models = camera_models
        self.workdir = workdir

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Camera Model Overrides"))

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Key", "Parameter", "Value"])
        layout.addWidget(self.table)

        save_button = QPushButton("Save Changes")
        save_button.clicked.connect(self.save_changes)
        layout.addWidget(save_button)

        self.setLayout(layout)

        self.load_camera_models()

    def load_camera_models(self):
        """カメラモデルをテーブルにロードする"""
        self.table.setRowCount(0)
        row = 0
        for key, params in self.camera_models.items():
            # key = 'Perspective' などカメラモデル名 (例: "Spherical")
            # params = { 'projection_type': 'perspective', 'width': ..., etc. }
            for param, value in params.items():
                self.table.insertRow(row)
                # 1列目(Key)
                self.table.setItem(row, 0, QTableWidgetItem(key))

                # 2列目(Parameter)
                self.table.setItem(row, 1, QTableWidgetItem(param))

                # 3列目(Value) - projection_type だけはComboBoxに
                if param == "projection_type":
                    combo = QComboBox()
                    combo.addItems(["Perspective", "Spherical"])
                    # 今の値に合わせて選択状態を変更
                    if str(value) in ["Perspective", "Spherical"]:
                        combo.setCurrentText(str(value))
                    else:
                        # もし「登録されていない値」だったら先頭に追加して選択
                        combo.insertItem(0, str(value))
                        combo.setCurrentIndex(0)
                    self.table.setCellWidget(row, 2, combo)
                else:
                    # それ以外はテキスト表示
                    self.table.setItem(row, 2, QTableWidgetItem(str(value)))

                row += 1

    def save_changes(self):
        """カメラモデルの変更を camera_models_overrides.json に保存"""
        updated_models = {}
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            param_item = self.table.item(row, 1)

            if not key_item or not param_item:
                continue

            key = key_item.text()
            param = param_item.text()

            # ValueがComboBoxかどうか判定
            cell_widget = self.table.cellWidget(row, 2)
            if isinstance(cell_widget, QComboBox):
                # ComboBoxなら現在のテキストを取得
                value = cell_widget.currentText()
            else:
                # 普通のQTableWidgetItemならテキストを取得
                value_item = self.table.item(row, 2)
                if value_item:
                    value = value_item.text()
                else:
                    value = ""

            # 数値ならfloat/intに変換して格納
            # ※ projection_type は文字列なので通常は変換されない
            try:
                # '.' を含む場合はfloat、それ以外はint
                if '.' in value:
                    num = float(value)
                    value = num
                else:
                    num = int(value)
                    value = num
            except ValueError:
                # 数値変換できなかった場合は文字列のまま
                pass

            if key not in updated_models:
                updated_models[key] = {}
            updated_models[key][param] = value

        # JSON 書き込み
        overrides_path = os.path.join(self.workdir, "camera_models_overrides.json")
        with open(overrides_path, "w") as f:
            json.dump(updated_models, f, indent=4)

        QMessageBox.information(self, "Success", "Camera models saved successfully!")
        self.accept()


def load_camera_models(workdir):
    """camera_models.json をベースに camera_models_overrides.json を適用する"""
    camera_models_path = os.path.join(workdir, "camera_models.json")
    overrides_path = os.path.join(workdir, "camera_models_overrides.json")

    if os.path.exists(camera_models_path):
        with open(camera_models_path, "r") as f:
            base_models = json.load(f)
    else:
        base_models = {}

    if os.path.exists(overrides_path):
        with open(overrides_path, "r") as f:
            overrides = json.load(f)
    else:
        overrides = {}

    merged_models = base_models.copy()
    for key, params in overrides.items():
        if key in merged_models:
            merged_models[key].update(params)
        else:
            merged_models[key] = params

    return merged_models


def open_camera_model_editor(main_app):
    """カメラモデルエディターを開く"""
    dialog = CameraModelEditor(main_app.camera_models, main_app.workdir, parent=main_app)
    if dialog.exec_():
        main_app.camera_models = load_camera_models(main_app.workdir)
        
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
        self.matching_tab_initialized = False

        self.feature_extractor = None  # Placeholder for FeatureExtractor instance
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
        self.camera_image_tree.setFixedWidth(250)
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
        self.camera_image_tree.setFixedWidth(250)
        layout.addWidget(self.camera_image_tree)

        if self.workdir is None:
            QMessageBox.warning(self, "Error", "Work directory is not set.")
            return

        if self.mask_manager is None:
            # Initialize MaskManager
            sam2_dir = self.get_sam2_directory()
            checkpoint_path = os.path.join(sam2_dir, "checkpoints", "sam2.1_hiera_large.pt")
            config_path = os.path.join("configs", "sam2.1", "sam2.1_hiera_l.yaml")
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
        self.camera_image_tree.setFixedWidth(250)
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

    def init_matching_tab(self):
        """Initialize the Matching tab."""
        layout = QSplitter(Qt.Horizontal)

        # Left side: Tree of images (Column 1)
        self.camera_image_tree_left = QTreeWidget()
        self.camera_image_tree_left.setHeaderLabel("Cameras and Images - Left")
        self.camera_image_tree_left.setFixedWidth(250)
        layout.addWidget(self.camera_image_tree_left)

        # Right side: Tree of images (Column 2)
        self.camera_image_tree_right = QTreeWidget()
        self.camera_image_tree_right.setHeaderLabel("Cameras and Images - Right")
        self.camera_image_tree_right.setFixedWidth(250)
        layout.addWidget(self.camera_image_tree_right)

        # Center: Add MatchingTab widget
        self.matching_viewer = FeatureMatching(workdir=self.workdir, image_list=self.image_list)
        layout.addWidget(self.matching_viewer)

        # Set layout for matching tab
        self.matching_tab.setLayout(QVBoxLayout())
        self.matching_tab.layout().addWidget(layout)

        # Populate each tree with camera data
        self.populate_tree_with_camera_data(self.camera_image_tree_left)
        self.populate_tree_with_camera_data(self.camera_image_tree_right)

        # Connect signals for image selection
        self.camera_image_tree_left.itemClicked.connect(self.on_image_selected_left)
        self.camera_image_tree_right.itemClicked.connect(self.on_image_selected_right)
    
    def init_reconstruct_tab(self):
        """Initialize the Reconstruct tab."""
        layout = QSplitter(Qt.Horizontal)

        # Left side: Tree of images grouped by camera
        self.camera_image_tree = QTreeWidget()
        self.camera_image_tree.setHeaderLabel("Cameras and Images")
        self.camera_image_tree.setFixedWidth(250)
        layout.addWidget(self.camera_image_tree)

        # Right side: PointCloudVisualizer widget
        self.reconstruction_viewer = Reconstruction(self.workdir)
        layout.addWidget(self.reconstruction_viewer)

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
            elif self.reconstruction_viewer:
                self.reconstruction_viewer.update_visualization()

        elif tab_name == "Features":
            # Featureタブの初期化処理
            if not self.features_tab_initialized:
                self.init_features_tab()
                self.features_tab_initialized = True

        elif tab_name == "Matching":
            # Featureタブの初期化処理
            if not self.matching_tab_initialized:
                self.init_matching_tab()
                self.matching_tab_initialized = True

    def show_start_dialog(self):
        """起動時にワークディレクトリを選択し、カメラモデルを編集する"""
        self.workdir = QFileDialog.getExistingDirectory(self, "Select Workdir")
        if not self.workdir:
            QMessageBox.warning(self, "Error", "No workdir selected. Exiting.")
            sys.exit(1)

        self.camera_models = load_camera_models(self.workdir)
        open_camera_model_editor(self)
        self.load_workdir()

    def load_workdir(self):
        """Load the work directory and initialize MaskManager."""
        if not self.workdir:  # もし workdir が未設定なら選択ダイアログを開く
            self.workdir = QFileDialog.getExistingDirectory(self, "Select Workdir")
            if not self.workdir:
                return

        # images フォルダの確認
        img_dir = os.path.join(self.workdir, "images")
        exif_dir = os.path.join(self.workdir, "exif")
        if not os.path.exists(img_dir):
            QMessageBox.warning(self, "Error", "images folder does not exist.")
            return
        else:
            try:
                dataset = DataSet(self.workdir)

                config_src = "config/config.yaml"
                config_dest = os.path.join(self.workdir, "config.yaml")
                shutil.copy(config_src, config_dest)
                from opensfm.actions import extract_metadata
                extract_metadata.run_dataset(dataset)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to extract metadata or copy config.yaml: {e}")
                return

            # masks フォルダの作成または確認
            mask_dir = os.path.join(self.workdir, "masks")
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir, exist_ok=True)

            # 画像リストの作成
            self.image_list = [
                f for f in os.listdir(img_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

            if self.image_list:
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
        """EXIF データをテーブルに表示し、overrides の情報があれば適用する"""
        self.exif_table.setRowCount(0)
        self.exif_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.exif_table.clearContents()

        # もし camera_models_overrides.json にカメラの設定があれば適用
        camera_name = exif_data.get("camera", "Unknown Camera")
        overrides = self.camera_models.get(camera_name, {})

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

            # EXIF データの値を取得し、overrides があれば適用
            value = overrides.get(field, exif_data.get(field, "N/A"))
            
            # JSON の辞書データを文字列に変換
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
        image_name = item.text(0)
        if self.feature_extractor:
            self.feature_extractor.load_image_by_name(image_name)
        else:
            QMessageBox.warning(self, "Error", "Feature extractor is not initialized.")

    def handle_camera_image_tree_click(self, item, column):
        """Handle single click event for camera_image_tree and pass image name to PointCloudVisualizer."""
        if item.childCount() == 0 and item.parent() is not None:
            image_name = item.text(0)
            self.reconstruction_viewer.on_camera_image_tree_click(image_name)

    def handle_camera_image_tree_double_click(self, item, column):
        """Handle double click event for camera_image_tree and move to the camera position."""
        if item.childCount() == 0 and item.parent() is not None:
            image_name = item.text(0)
            self.reconstruction_viewer.on_camera_image_tree_double_click(image_name)

    def on_image_selected_left(self, item, column):
        """Handle image selection in the left camera tree."""
        if item.parent():
            image_name = item.text(0)
            self.matching_viewer.load_image_by_name(image_name, position="left")

    def on_image_selected_right(self, item, column):
        """Handle image selection in the right camera tree."""
        if item.parent():
            image_name = item.text(0)
            self.matching_viewer.load_image_by_name(image_name, position="right")

    def get_sam2_directory(self):
        """
        get sam2 install directory
        """
        module_name = 'sam2'
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            raise ModuleNotFoundError(f"{module_name} is not installed.")
        return os.path.dirname(os.path.dirname(spec.origin))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
