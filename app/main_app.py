# main_app.py
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QToolBar, QAction, QFileDialog, QMessageBox, QDialog, QVBoxLayout, QPushButton, QLabel, QWidget
)
from PyQt5.QtCore import Qt, QTimer
from opensfm import commands
from app.mask_manager import MaskManager  # Import the updated MaskManager class
from opensfm import commands
from opensfm.dataset import DataSet
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

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM2 Mask Creator")
        self.setGeometry(100, 100, 1000, 800)

        # Set up toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Initialize MaskManager placeholder
        self.mask_manager = None

        # Set an empty central widget to prevent the window from collapsing
        self.setCentralWidget(QWidget())

        # After showing the main window, display the start dialog
        QTimer.singleShot(0, self.show_start_dialog)

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
        workdir = QFileDialog.getExistingDirectory(self, "Select Workdir")
        if not workdir:
            return

        # images folder check
        img_dir = os.path.join(workdir, "images")
        if not os.path.exists(img_dir):
            QMessageBox.warning(self, "Error", "images folder does not exist.")
            return
        else:
            try:
                dataset = DataSet(workdir)
                from opensfm.actions import extract_metadata
                extract_metadata.run_dataset(dataset)
                QMessageBox.information(self, "Success", "Metadata extracted successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to extract metadata: {e}")
                return

            # Create or check masks folder
            mask_dir = os.path.join(workdir, "masks")
            if new_project:
                # For new projects, create masks folder
                os.makedirs(mask_dir, exist_ok=True)
            else:
                # For existing data, check if masks folder exists
                if not os.path.exists(mask_dir):
                    QMessageBox.warning(self, "Error", "masks folder does not exist in the selected directory.")
                    return

            image_list = [
                f for f in os.listdir(img_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if image_list:
                # Initialize MaskManager
                checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
                config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
                self.mask_manager = MaskManager(
                    checkpoint_path, config_path, mask_dir, img_dir
                )
                self.setCentralWidget(self.mask_manager)
                QMessageBox.information(self, "Success", "Workdir loaded successfully.")
            else:
                QMessageBox.warning(self, "Error", "No images found in the images folder.")
                return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
