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
        #self.init_masks_tab()
        # Features, Matching, Gsplat tabs are placeholders for now
        #self.init_reconstruct_tab()

    def init_images_tab(self):
        """Initialize the Images tab."""
        layout = QSplitter(Qt.Horizontal)

        # Left side: Tree of images grouped by camera
        self.images_tree = QTreeWidget()
        self.images_tree.setHeaderHidden(True)
        layout.addWidget(self.images_tree)

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

        self.images_tab.setLayout(QVBoxLayout())
        self.images_tab.layout().addWidget(layout)

        # Connect signals
        self.images_tree.itemClicked.connect(self.display_image_and_exif)

    def on_tab_changed(self, index):
        """Handle actions when a tab is changed."""
        tab_name = self.tab_widget.tabText(index)
        if tab_name == "Masks":
            if not self.masks_tab_initialized:
                self.init_masks_tab()
                self.masks_tab_initialized = True
            else:
                if self.mask_manager.predictor is None:
                    self.mask_manager.init_sam_model()
        else:
            if self.masks_tab_initialized and self.mask_manager.predictor is not None:
                self.mask_manager.unload_sam_model()
                    
    def init_masks_tab(self):
        """Initialize the Masks tab."""
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

        layout = QSplitter(Qt.Horizontal)

        # Left side: Tree of masks grouped by camera
        self.masks_tree = QTreeWidget()
        self.masks_tree.setHeaderHidden(True)
        layout.addWidget(self.masks_tree)

        # Right side: MaskManager widget
        self.mask_manager_widget = MaskManagerWidget(parent=self)
        layout.addWidget(self.mask_manager_widget)

        # Set new layout for masks_tab
        self.masks_tab.setLayout(QVBoxLayout())
        self.masks_tab.layout().addWidget(layout)

        # Now, call populate_mask_list() after self.masks_tree is initialized
        self.populate_mask_list()

        # Connect signals
        self.masks_tree.itemClicked.connect(self.display_mask)

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
                # For new projects, create masks folder
                os.makedirs(mask_dir, exist_ok=True)
            else:
                # For existing data, check if masks folder exists
                if not os.path.exists(mask_dir):
                    QMessageBox.warning(self, "Error", "masks folder does not exist in the selected directory.")
                    return

            # Set self.image_list here
            self.image_list = [
                f for f in os.listdir(img_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if self.image_list:
                # Initialize MaskManager will be done when Masks tab is opened
                # Update image tree
                self.populate_image_tree()
                # Remove or comment out the following line:
                # self.populate_mask_list()
                QMessageBox.information(self, "Success", "Workdir loaded successfully.")
            else:
                QMessageBox.warning(self, "Error", "No images found in the images folder.")
                return

    def populate_image_tree(self):
        """Populate the images tree in the Images tab, grouped by camera."""
        self.images_tree.clear()
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
            camera_item = QTreeWidgetItem(self.images_tree)
            camera_item.setText(0, camera)
            for img in images:
                img_item = QTreeWidgetItem(camera_item)
                img_item.setText(0, img)

    def populate_mask_list(self):
        """Populate the masks tree in the Masks tab, grouped by camera."""
        self.masks_tree.clear()
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
            camera_item = QTreeWidgetItem(self.masks_tree)
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
