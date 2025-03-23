# reconstruct_tab.py

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QLabel, QTreeWidget, QMessageBox
)
from PyQt5.QtCore import Qt

from app.base_tab import BaseTab

class ReconstructTab(BaseTab):
    """Reconstruct tab implementation"""
    def __init__(self, workdir=None, image_list=None, parent=None):
        super().__init__(workdir, image_list, parent)
        self.reconstruction_viewer = None
        self.camera_image_tree = None
    
    def get_tab_name(self):
        return "Reconstruct"
    
    def initialize(self):
        """Initialize the Reconstruct tab"""
        if not self.workdir:
            QMessageBox.warning(self, "Error", "Work directory is not set.")
            return
        
        layout = self.create_horizontal_splitter()
        
        # Left side: Tree of images grouped by camera
        self.camera_image_tree = QTreeWidget()
        self.camera_image_tree.setHeaderLabel("Cameras and Images")
        self.camera_image_tree.setFixedWidth(250)
        layout.addWidget(self.camera_image_tree)
        
        try:
            # Import Reconstruction here to avoid circular imports
            from app.point_cloud_visualizer import Reconstruction
            
            # Right side: Reconstruction (PointCloudVisualizer) widget
            self.reconstruction_viewer = Reconstruction(self.workdir)
            layout.addWidget(self.reconstruction_viewer)
            
            # Set layout for reconstruct tab
            self._layout.addWidget(layout)
            
            # Setup camera image tree with click and double click event handlers
            self.setup_camera_image_tree(self.camera_image_tree)
            self.camera_image_tree.itemClicked.connect(self.handle_camera_image_tree_click)
            self.camera_image_tree.itemDoubleClicked.connect(self.handle_camera_image_tree_double_click)
            
            # Set stretch factors
            layout.setStretchFactor(0, 1)  # Left side (image tree)
            layout.setStretchFactor(1, 4)  # Right side (reconstruction viewer)
            
            self.is_initialized = True
            
        except Exception as e:
            error_message = f"Failed to initialize Reconstruction: {str(e)}"
            QMessageBox.critical(self, "Error", error_message)
            placeholder = QLabel("Reconstruction could not be initialized. See error message.")
            placeholder.setAlignment(Qt.AlignCenter)
            layout.addWidget(placeholder)
            self._layout.addWidget(layout)
    
    def handle_camera_image_tree_click(self, item, column):
        """Handle single click event for camera_image_tree"""
        if item.childCount() == 0 and item.parent() is not None:
            image_name = item.text(0)
            if self.reconstruction_viewer:
                self.reconstruction_viewer.on_camera_image_tree_click(image_name)
    
    def handle_camera_image_tree_double_click(self, item, column):
        """Handle double click event for camera_image_tree"""
        if item.childCount() == 0 and item.parent() is not None:
            image_name = item.text(0)
            if self.reconstruction_viewer:
                self.reconstruction_viewer.on_camera_image_tree_double_click(image_name)
    
    def refresh(self):
        """Refresh the tab content"""
        if self.is_initialized:
            # Remove old widgets
            for i in reversed(range(self._layout.count())): 
                self._layout.itemAt(i).widget().setParent(None)
            
            # Reinitialize
            self.is_initialized = False
            self.initialize()
    
    def on_tab_activated(self):
        """Called when tab is activated"""
        super().on_tab_activated()
        # Update visualization when tab is activated
        if self.is_initialized and self.reconstruction_viewer:
            self.reconstruction_viewer.update_visualization()