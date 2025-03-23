# images_tab.py

import os
import json
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QTreeWidget, QTableWidget, QTableWidgetItem, 
    QSizePolicy, QMessageBox, QApplication, QDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

from app.base_tab import BaseTab
from app.camera_models import CameraModelManager
from app.image_processing import ImageProcessor, ResolutionDialog, ExifExtractProgressDialog

class ImagesTab(BaseTab):
    """Images tab implementation"""
    def __init__(self, workdir=None, image_list=None, parent=None):
        super().__init__(workdir, image_list, parent)
        self.image_viewer = None
        self.exif_table = None
        self.camera_image_tree = None
        self.camera_model_manager = None
        self.image_processor = None
        
        # Always set up the basic UI structure
        self.setup_basic_ui()
        
        # Initialize managers if workdir is available
        if self.workdir:
            try:
                self.camera_model_manager = CameraModelManager(self.workdir)
                self.image_processor = ImageProcessor(self.workdir)
            except Exception as e:
                print(f"Error initializing managers: {e}")
    
    def get_tab_name(self):
        return "Images"
    
    def setup_basic_ui(self):
        """Set up the basic UI structure without data initialization"""
        layout = self.create_horizontal_splitter()
        
        # Left side: Camera and image tree
        self.camera_image_tree = QTreeWidget()
        self.camera_image_tree.setHeaderLabel("Cameras and Images")
        self.camera_image_tree.setFixedWidth(250)
        layout.addWidget(self.camera_image_tree)
        
        # Right side: Image viewer, EXIF data, buttons
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Image viewer
        self.image_viewer = QLabel("Image Viewer")
        self.image_viewer.setAlignment(Qt.AlignCenter)
        self.image_viewer.setMinimumHeight(300)
        right_layout.addWidget(self.image_viewer, stretch=3)
        
        # EXIF data table
        self.exif_table = QTableWidget()
        self.exif_table.setColumnCount(2)
        self.exif_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.exif_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.exif_table, stretch=2)
        
        # Buttons
        button_widget = QWidget()
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(5, 5, 5, 5)
        button_layout.setSpacing(10)
        
        self.camera_model_button = QPushButton("Edit Camera Models")
        self.camera_model_button.clicked.connect(self.open_camera_model_editor)
        
        self.resize_button = QPushButton("Change Resolution")
        self.resize_button.clicked.connect(self.resize_images_in_folder)
        
        self.restore_button = QPushButton("Restore Images")
        self.restore_button.clicked.connect(self.restore_original_images)
        
        # Add stretch to center the buttons
        button_layout.addStretch(1)
        button_layout.addWidget(self.camera_model_button)
        button_layout.addWidget(self.resize_button)
        button_layout.addWidget(self.restore_button)
        button_layout.addStretch(1)
        
        button_widget.setLayout(button_layout)
        
        # Make button widget expand horizontally
        button_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        right_layout.addWidget(button_widget, stretch=0)
        
        right_widget.setLayout(right_layout)
        layout.addWidget(right_widget)
        
        layout.setStretchFactor(0, 1)
        layout.setStretchFactor(1, 4)
        
        self._layout.addWidget(layout)
        
        # Connect signals
        self.camera_image_tree.itemClicked.connect(self.display_image_and_exif)
    
    def initialize(self):
        """Initialize the Images tab with data"""
        # Basic UI is already set up in __init__, so we only need to initialize data
        if self.workdir:
            # Initialize managers if not already initialized
            if self.camera_model_manager is None:
                try:
                    self.camera_model_manager = CameraModelManager(self.workdir)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to initialize camera model manager: {e}")
            
            if self.image_processor is None:
                try:
                    self.image_processor = ImageProcessor(self.workdir)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to initialize image processor: {e}")
            
            # Populate the camera tree
            try:
                self.setup_camera_image_tree(self.camera_image_tree)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to populate camera tree: {e}")
        
        self.is_initialized = True
    
    def display_image_and_exif(self, item, column):
        """Display the selected image and its EXIF data"""
        if not self.is_initialized:
            # Make sure we're initialized before attempting to display data
            self.initialize()
            
        if item.childCount() == 0 and item.parent() is not None:
            image_name = item.text(0)
            image_path = os.path.join(self.workdir, "images", image_name)
            exif_path = os.path.join(self.workdir, "exif", image_name + '.exif')
            
            # Display image
            if os.path.exists(image_path):
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
    
    def display_exif_data(self, exif_data):
        """Display EXIF data in the table, applying overrides if available"""
        self.exif_table.setRowCount(0)
        self.exif_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.exif_table.clearContents()
        
        # Apply camera_models_overrides.json settings if available
        camera_name = exif_data.get("camera", "Unknown Camera")
        overrides = {}
        if self.camera_model_manager:
            try:
                camera_models = self.camera_model_manager.get_camera_models()
                overrides = camera_models.get(camera_name, {})
            except Exception as e:
                print(f"Error getting camera models: {e}")
        
        # Define fields to display and their order
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
            
            # Get value from EXIF data or overrides
            value = overrides.get(field, exif_data.get(field, "N/A"))
            
            # Convert dictionary to string
            if isinstance(value, dict):
                value = json.dumps(value)
            elif isinstance(value, float):
                value = f"{value:.2f}"
            
            value_item = QTableWidgetItem(str(value))
            self.exif_table.setItem(i, 0, key_item)
            self.exif_table.setItem(i, 1, value_item)
    
    def open_camera_model_editor(self):
        """Open the camera model editor dialog"""
        if not self.is_initialized:
            self.initialize()
        
        # Check if camera model manager exists, if not, try to create it
        if self.camera_model_manager is None and self.workdir:
            try:
                self.camera_model_manager = CameraModelManager(self.workdir)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to initialize camera model manager: {e}")
                return
            
        if self.camera_model_manager:
            # Ensure camera_models.json exists
            camera_models_path = os.path.join(self.workdir, "camera_models.json")
            if not os.path.exists(camera_models_path):
                # Create a default camera model if needed
                try:
                    with open(camera_models_path, 'w') as f:
                        default_model = {
                            "Perspective": {
                                "projection_type": "perspective",
                                "width": 1920,
                                "height": 1080,
                                "focal_ratio": 1.0
                            }
                        }
                        json.dump(default_model, f, indent=4)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to create default camera model: {e}")
                    return
                
                # Reload the camera model manager
                try:
                    self.camera_model_manager = CameraModelManager(self.workdir)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to reinitialize camera model manager: {e}")
                    return
            
            # Now open the editor
            try:
                self.camera_model_manager.open_camera_model_editor(parent=self)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open camera model editor: {e}")
        else:
            QMessageBox.warning(self, "Error", "Camera Model Manager is not initialized and cannot be created.")
    
    def resize_images_in_folder(self):
        """Open dialog to resize images"""
        if not self.is_initialized:
            self.initialize()
            
        # Check if image processor exists, if not, try to create it
        if self.image_processor is None and self.workdir:
            try:
                self.image_processor = ImageProcessor(self.workdir)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to initialize image processor: {e}")
                return
        
        if not self.image_processor:
            QMessageBox.warning(self, "Error", "Image Processor is not initialized and cannot be created.")
            return
        
        # Get dimensions of a sample image
        width, height = self.image_processor.get_sample_image_dimensions()
        if width is None or height is None:
            QMessageBox.warning(self, "Error", "No images found to determine default resolution.")
            return
        
        # Show resolution dialog
        dialog = ResolutionDialog(width, height, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        
        method, value = dialog.get_values()
        
        # Show progress dialog
        progress_dialog = ExifExtractProgressDialog("Resizing images...", self)
        progress_dialog.show()
        QApplication.processEvents()
        
        try:
            # Resize images
            self.image_processor.resize_images(method, value)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error during resizing: {e}")
        finally:
            progress_dialog.close()
        
        QMessageBox.information(self, "Completed", "Images resized successfully!")
    
    def restore_original_images(self):
        """Restore original images from backup"""
        if not self.is_initialized:
            self.initialize()
            
        # Check if image processor exists, if not, try to create it
        if self.image_processor is None and self.workdir:
            try:
                self.image_processor = ImageProcessor(self.workdir)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to initialize image processor: {e}")
                return
        
        if not self.image_processor:
            QMessageBox.warning(self, "Error", "Image Processor is not initialized and cannot be created.")
            return
        
        result = self.image_processor.restore_original_images()
        if result:
            QMessageBox.information(self, "Restored", "Original images restored successfully!")
        else:
            QMessageBox.warning(self, "Error", "No original backup images found.")
    
    def refresh(self):
        """Refresh the tab contents"""
        if self.is_initialized:
            if self.camera_model_manager:
                try:
                    self.camera_model_manager = CameraModelManager(self.workdir)
                except Exception as e:
                    print(f"Error refreshing camera model manager: {e}")
            
            if self.image_processor:
                try:
                    self.image_processor = ImageProcessor(self.workdir)
                except Exception as e:
                    print(f"Error refreshing image processor: {e}")
            
            # Refresh camera image tree
            if self.camera_image_tree:
                try:
                    self.setup_camera_image_tree(self.camera_image_tree)
                except Exception as e:
                    print(f"Error refreshing camera image tree: {e}")
