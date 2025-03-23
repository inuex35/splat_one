# masks_tab.py

import os
import sys
import importlib.util
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel,
    QTreeWidget, QTreeWidgetItem, QMessageBox
)
from PyQt5.QtCore import Qt

from app.base_tab import BaseTab

class MaskManagerWidget(QWidget):
    def __init__(self, mask_manager, parent=None):
        super().__init__(parent)
        self.mask_manager = mask_manager
        layout = QVBoxLayout()
        layout.addWidget(self.mask_manager)
        self.setLayout(layout)

class MasksTab(BaseTab):
    """Masks tab implementation"""
    def __init__(self, workdir=None, image_list=None, parent=None):
        super().__init__(workdir, image_list, parent)
        self.mask_manager = None
        self.mask_manager_widget = None
        self.camera_image_tree = None
        
    def get_tab_name(self):
        return "Masks"
    
    def initialize(self):
        """Initialize the Masks tab"""
        if not self.workdir:
            QMessageBox.warning(self, "Error", "Work directory is not set.")
            return
            
        layout = self.create_horizontal_splitter()
            
        # Left side: Tree of images grouped by camera
        self.camera_image_tree = QTreeWidget()
        self.camera_image_tree.setHeaderLabel("Cameras and Images")
        self.camera_image_tree.setFixedWidth(250)
        layout.addWidget(self.camera_image_tree)
            
        # Initialize MaskManager
        try:
            sam2_dir = self.get_sam2_directory()
            checkpoint_path = os.path.join(sam2_dir, "checkpoints", "sam2.1_hiera_large.pt")
            config_path = os.path.join("configs", "sam2.1", "sam2.1_hiera_l.yaml")
            mask_dir = os.path.join(self.workdir, "masks")
            img_dir = os.path.join(self.workdir, "images")
                
            # Ensure mask directory exists
            os.makedirs(mask_dir, exist_ok=True)
                
            # Import MaskManager here to avoid importing when not needed
            from app.mask_manager import MaskManager
                
            self.mask_manager = MaskManager(
                checkpoint_path, config_path, mask_dir, img_dir, self.image_list
            )
                
            # Right side: MaskManager widget
            self.mask_manager_widget = MaskManagerWidget(self.mask_manager)
            layout.addWidget(self.mask_manager_widget)
                
            # Set stretch factors
            layout.setStretchFactor(0, 1)  # Left side (image tree)
            layout.setStretchFactor(1, 4)  # Right side (MaskManager)
                
            # Set layout for mask tab
            self._layout.addWidget(layout)
                
            # Populate the camera image tree
            self.setup_camera_image_tree(self.camera_image_tree, self.display_mask)
                
            self.is_initialized = True
                
        except Exception as e:
            error_message = f"Failed to initialize MaskManager: {str(e)}"
            QMessageBox.critical(self, "Error", error_message)
            placeholder = QLabel("MaskManager could not be initialized. See error message.")
            placeholder.setAlignment(Qt.AlignCenter)
            layout.addWidget(placeholder)
            self._layout.addWidget(layout)
    
    def display_mask(self, item, column):
        """Display the mask for the selected image"""
        if item.childCount() == 0 and item.parent() is not None:
            image_name = item.text(0)
            if self.mask_manager is not None:
                self.mask_manager.load_image_by_name(image_name)
    
    def on_tab_activated(self):
        """Called when tab is activated"""
        super().on_tab_activated()
        # Load SAM model when tab is activated
        if self.mask_manager and hasattr(self.mask_manager, 'init_sam_model'):
            self.mask_manager.init_sam_model()
    
    def on_tab_deactivated(self):
        """Called when tab is deactivated"""
        super().on_tab_deactivated()
        # Unload SAM model when tab is deactivated to save memory
        if self.mask_manager and hasattr(self.mask_manager, 'unload_sam_model'):
            self.mask_manager.unload_sam_model()
    
    def refresh(self):
        """Refresh the tab content"""
        # Reinitialize the tab if initialized
        if self.is_initialized:
            # Remove old widgets
            for i in reversed(range(self._layout.count())): 
                self._layout.itemAt(i).widget().setParent(None)
            
            # Reinitialize
            self.is_initialized = False
            self.on_tab_activated()
    
    def get_sam2_directory(self):
        """Get sam2 install directory"""
        module_name = 'sam2'
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            raise ModuleNotFoundError(f"{module_name} is not installed.")
        return os.path.dirname(os.path.dirname(spec.origin))