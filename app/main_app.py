# main_app.py

import sys
import os
import json
import shutil
import importlib.util
from PyQt5.QtWidgets import (
    QMainWindow, QToolBar, QFileDialog, QMessageBox,
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFormLayout,
    QComboBox, QLineEdit, QCheckBox, QDialogButtonBox, QApplication
)
from PyQt5.QtCore import Qt, QTimer
from pathlib import Path
from argparse import Namespace
from mapillary_tools.commands.video_process import Command as VideoProcessCommand

# Import managers and utilities
from app.tab_manager import TabManager
from app.camera_models import CameraModelManager
from app.image_processing import ImageProcessor, ExifExtractProgressDialog

# Import all tab classes
from app.tabs import ImagesTab, MasksTab, FeaturesTab, MatchingTab, ReconstructTab, GsplatTab

class VideoProcessDialog(QDialog):
    """Dialog for video processing settings"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Processing Parameters")
        self.setupUI()
    
    def setupUI(self):
        form_layout = QFormLayout()
        
        self.import_path_input = QLineEdit()
        form_layout.addRow("Import path (leave empty for default):", self.import_path_input)
        
        # Sampling method selection
        self.sampling_method_combo = QComboBox()
        self.sampling_method_combo.addItems(["Interval", "Distance"])
        form_layout.addRow("Sampling method:", self.sampling_method_combo)
        
        # Interval input (default 0.5 sec)
        self.interval_input = QLineEdit("0.5")
        form_layout.addRow("Sample interval (seconds):", self.interval_input)
        
        # Distance input
        self.distance_input = QLineEdit("5")  # default distance 5 meters
        self.distance_input.setDisabled(True)  # Default disabled
        form_layout.addRow("Sample distance (meters):", self.distance_input)
        
        # Connect signals
        self.sampling_method_combo.currentIndexChanged.connect(self.toggle_sampling_inputs)
        
        # Geotag source
        self.geotag_source_combo = QComboBox()
        self.geotag_source_combo.addItem("camm")  # Default 
        self.geotag_source_combo.addItems([
            "video", "camm", "gopro", "blackvue", "gpx",
            "nmea", "exiftool_xml", "exiftool_runtime"
        ])
        form_layout.addRow("Geotag source (optional):", self.geotag_source_combo)
        
        # Geotag source path
        self.geotag_source_path_input = QLineEdit()
        form_layout.addRow("Geotag source path (if needed):", self.geotag_source_path_input)
        
        # Interpolation offset
        self.interpolation_offset_input = QLineEdit("0")
        form_layout.addRow("Interpolation offset time (seconds):", self.interpolation_offset_input)
        
        # Use GPX checkbox
        self.interpolation_use_gpx_checkbox = QCheckBox()
        self.interpolation_use_gpx_checkbox.setChecked(True)
        form_layout.addRow("Use GPX start time for interpolation:", self.interpolation_use_gpx_checkbox)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        form_layout.addRow(button_box)
        
        self.setLayout(form_layout)
    
    def toggle_sampling_inputs(self):
        """Toggle between interval and distance inputs"""
        if self.sampling_method_combo.currentText() == "Distance":
            self.distance_input.setDisabled(False)
            self.interval_input.setDisabled(True)
        else:
            self.distance_input.setDisabled(True)
            self.interval_input.setDisabled(False)
    
    def get_sampling_values(self):
        """Get sampling method and value"""
        method = self.sampling_method_combo.currentText()
        if method == "Distance":
            distance = float(self.distance_input.text().strip())
            interval = -1
        else:
            interval = float(self.interval_input.text().strip())
            distance = -1
        return method, interval, distance
    
    def get_values(self):
        """Get all dialog values"""
        import_path = self.import_path_input.text().strip()
        method, interval, distance = self.get_sampling_values()
        geotag_source = self.geotag_source_combo.currentText() or None
        geotag_source_path = self.geotag_source_path_input.text().strip() or None
        offset_time = float(self.interpolation_offset_input.text().strip() or 0)
        use_gpx = self.interpolation_use_gpx_checkbox.isChecked()
        
        return {
            "import_path": import_path,
            "method": method,
            "interval": interval,
            "distance": distance,
            "geotag_source": geotag_source,
            "geotag_source_path": geotag_source_path,
            "offset_time": offset_time,
            "use_gpx": use_gpx
        }


class MainApp(QMainWindow):
    """Main application class"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPLAT_ONE")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize attributes
        self.workdir = None
        self.image_list = []
        self.tab_manager = None
        self.camera_model_manager = None
        self.image_processor = None
        
        # Set up the UI
        self.setup_ui()
        
        # After showing the main window, display the start dialog
        QTimer.singleShot(0, self.show_start_dialog)
    
    def setup_ui(self):
        """Set up the user interface"""
        # Set up toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # Set up tab manager
        self.tab_manager = TabManager(self)
        self.setCentralWidget(self.tab_manager)
        
        # Register tabs
        self.register_tabs()
    
    def register_tabs(self):
        """Register application tabs"""
        # Register all tabs with appropriate parameters
        self.tab_manager.register_tab(ImagesTab, workdir=self.workdir, image_list=self.image_list)
        self.tab_manager.register_tab(MasksTab, workdir=self.workdir, image_list=self.image_list)
        self.tab_manager.register_tab(FeaturesTab, workdir=self.workdir, image_list=self.image_list)
        self.tab_manager.register_tab(MatchingTab, workdir=self.workdir, image_list=self.image_list)
        self.tab_manager.register_tab(ReconstructTab, workdir=self.workdir, image_list=self.image_list)
        self.tab_manager.register_tab(GsplatTab, workdir=self.workdir, image_list=self.image_list)
    
    def show_start_dialog(self):
        """Prompt user at startup to select processing type"""
        choice = QMessageBox.question(
            self,
            "Select Input Type",
            "Do you want to process a video file?\n"
            "(Choose 'No' to select an image folder instead.)",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if choice == QMessageBox.Yes:
            # Video processing
            self.process_video()
        else:
            # Image folder selection
            self.select_image_folder()
    
    def select_image_folder(self):
        """Select and process an image folder"""
        self.workdir = QFileDialog.getExistingDirectory(self, "Select Work Directory (Image Folder)")
        if not self.workdir:
            QMessageBox.warning(self, "Error", "Work directory not selected. Exiting.")
            sys.exit(1)
        
        # Ensure images directory exists
        images_output_dir = os.path.join(self.workdir, "images")
        os.makedirs(images_output_dir, exist_ok=True)
        
        # Copy images from root to images directory if needed
        for image_file in os.listdir(self.workdir):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(self.workdir, image_file)
                dst = os.path.join(images_output_dir, image_file)
                if not os.path.exists(dst):
                    shutil.copy(src, dst)
        
        # Load the work directory
        self.load_workdir()
    
    def process_video(self):
        """Process a video file"""
        # Select video file
        video_file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Videos (*.mp4 *.mov *.avi *.mkv)"
        )
        if not video_file_path:
            QMessageBox.warning(self, "Error", "Video file not selected. Exiting.")
            sys.exit(1)
        
        # Get video file info
        video_file_name = os.path.splitext(os.path.basename(video_file_path))[0]
        video_file_dir = os.path.splitext(video_file_path)[0]
        
        # Show parameter dialog
        dialog = VideoProcessDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            QMessageBox.warning(self, "Cancelled", "Video processing cancelled. Exiting.")
            sys.exit(1)
        
        # Get parameters
        params = dialog.get_values()
        import_path = params["import_path"] or os.path.join(video_file_dir, video_file_name)
        os.makedirs(import_path, exist_ok=True)
        
        # Show progress dialog
        progress_dialog = ExifExtractProgressDialog("Extracting frames and EXIF data from video...", self)
        progress_dialog.show()
        QApplication.processEvents()
        
        try:
            # Create arguments for video processing
            args = Namespace(
                video_import_path=Path(video_file_path),
                import_path=Path(import_path),
                video_sample_distance=params["distance"],
                video_sample_interval=params["interval"],
                filetypes={"image"},
                geotag_source=params["geotag_source"],
                video_geotag_source=None,
                video_geotag_source_path=params["geotag_source_path"],
                interpolation_offset_time=params["offset_time"],
                interpolation_use_gpx_start_time=params["use_gpx"],
                skip_process_errors=True,
            )
            
            # Process video
            command = VideoProcessCommand()
            command.run(vars(args))
            
            # Apply EXIF data from Mapillary JSON
            json_path = os.path.join(import_path, "mapillary_image_description.json")
            video_dir_name = os.path.splitext(os.path.basename(video_file_path))
            images_dir = os.path.join(import_path, video_dir_name[0] + video_dir_name[1])
            
            if os.path.exists(json_path) and os.path.exists(images_dir):
                # Create a new ImageProcessor specifically for this operation
                processor = ImageProcessor(import_path)
                processor.apply_exif_from_mapillary_json(json_path, images_dir)
            else:
                QMessageBox.warning(self, "Warning", "mapillary_image_description.json or images directory not found.")
            
        except Exception as e:
            QMessageBox.warning(self, "Processing Error", f"Error: {str(e)}")
            progress_dialog.close()
            return
        
        progress_dialog.close()
        
        # Set workdir and load it
        self.workdir = import_path
        self.load_workdir()
    
    def load_workdir(self):
        """Load the work directory"""
        if not self.workdir:
            self.workdir = QFileDialog.getExistingDirectory(self, "Select Workdir")
            if not self.workdir:
                return
        
        # Check for images directory
        img_dir = os.path.join(self.workdir, "images")
        exif_dir = os.path.join(self.workdir, "exif")
        os.makedirs(exif_dir, exist_ok=True)  # Ensure exif directory exists
        
        if not os.path.exists(img_dir):
            QMessageBox.warning(self, "Error", "images folder does not exist.")
            sys.exit(1)
        
        # Get image list
        self.image_list = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        # Extract EXIF data if needed
        exif_exists_for_all_images = (
            all(os.path.exists(os.path.join(exif_dir, f"{image}.exif")) for image in self.image_list)
            if self.image_list else False
        )
        
        if not exif_exists_for_all_images:
            progress_dialog = ExifExtractProgressDialog("Extracting EXIF data...", self)
            progress_dialog.show()
            QApplication.processEvents()
            
            try:
                from opensfm.dataset import DataSet
                dataset = DataSet(self.workdir)
                
                # Copy config file if needed
                config_src = "config/config.yaml"
                config_dest = os.path.join(self.workdir, "config.yaml")
                if os.path.exists(config_src):
                    shutil.copy(config_src, config_dest)
                else:
                    QMessageBox.warning(self, "Warning", f"Config file {config_src} not found.")
                
                # Extract metadata
                from opensfm.actions import extract_metadata
                extract_metadata.run_dataset(dataset)
                
                progress_dialog.close()
                
                # Create camera model manager and show editor
                self.camera_model_manager = CameraModelManager(self.workdir)
                self.camera_model_manager.open_camera_model_editor(parent=self)
                
            except Exception as e:
                progress_dialog.close()
                QMessageBox.warning(self, "Error", f"Failed to extract metadata: {e}")
        else:
            # Create camera model manager
            self.camera_model_manager = CameraModelManager(self.workdir)
        
        # Create image processor
        self.image_processor = ImageProcessor(self.workdir)
        
        # Create masks directory if needed
        mask_dir = os.path.join(self.workdir, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        
        # Update all tabs with new workdir and image list
        if self.tab_manager:
            self.tab_manager.update_all_tabs(workdir=self.workdir, image_list=self.image_list)
        
        if not self.image_list:
            QMessageBox.warning(self, "Error", "No images found in the images folder.")
