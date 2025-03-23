# features_tab.py

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QLabel, QTreeWidget, QMessageBox
)
from PyQt5.QtCore import Qt

from app.base_tab import BaseTab

class FeaturesTab(BaseTab):
    """Features tab implementation"""
    def __init__(self, workdir=None, image_list=None, parent=None):
        super().__init__(workdir, image_list, parent)
        self.feature_extractor = None
        self.camera_image_tree = None
    
    def get_tab_name(self):
        return "Features"
    
    def initialize(self):
        """Initialize the Features tab"""
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
            # Import FeatureExtractor here to avoid circular imports
            from app.feature_extractor import FeatureExtractor
            
            # Right side: FeatureExtractor widget
            self.feature_extractor = FeatureExtractor(self.workdir, self.image_list)
            layout.addWidget(self.feature_extractor)
            
            # Set layout for features tab
            self._layout.addWidget(layout)
            
            # Populate the tree with camera data
            self.setup_camera_image_tree(self.camera_image_tree, self.display_features_for_image)
            
            # Set stretch factors
            layout.setStretchFactor(0, 1)  # Left side (image tree)
            layout.setStretchFactor(1, 4)  # Right side (feature extractor)
            
            self.is_initialized = True
            
        except Exception as e:
            error_message = f"Failed to initialize FeatureExtractor: {str(e)}"
            QMessageBox.critical(self, "Error", error_message)
            placeholder = QLabel("FeatureExtractor could not be initialized. See error message.")
            placeholder.setAlignment(Qt.AlignCenter)
            layout.addWidget(placeholder)
            self._layout.addWidget(layout)
    
    def display_features_for_image(self, item, column):
        """Display features for the selected image"""
        if item.childCount() == 0 and item.parent() is not None:
            image_name = item.text(0)
            if self.feature_extractor and hasattr(self.feature_extractor, 'load_image_by_name'):
                self.feature_extractor.load_image_by_name(image_name)
    
    def refresh(self):
        """Refresh the tab content"""
        if self.is_initialized:
            # Remove old widgets
            for i in reversed(range(self._layout.count())): 
                self._layout.itemAt(i).widget().setParent(None)
            
            # Reinitialize
            self.is_initialized = False
            self.initialize()