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
        
        # Set up basic UI structure
        self.setup_basic_ui()
    
    def get_tab_name(self):
        return "Features"
    
    def setup_basic_ui(self):
        """Set up the basic UI structure"""
        layout = self.create_horizontal_splitter()
        
        # Left side: Tree of images grouped by camera
        self.camera_image_tree = QTreeWidget()
        self.camera_image_tree.setHeaderLabel("Cameras and Images")
        self.camera_image_tree.setFixedWidth(250)
        layout.addWidget(self.camera_image_tree)
        
        # Right side: Placeholder for feature extractor
        right_widget = QLabel("Feature Extractor will be displayed here.")
        right_widget.setAlignment(Qt.AlignCenter)
        layout.addWidget(right_widget)
        
        # Set stretch factors
        layout.setStretchFactor(0, 1)  # Left side (image tree)
        layout.setStretchFactor(1, 4)  # Right side (feature extractor)
        
        # Set layout for features tab
        self._layout.addWidget(layout)
        
        # Connect signals
        self.camera_image_tree.itemClicked.connect(self.display_features_for_image)
    
    def initialize(self):
        """Initialize the Features tab with data"""
        if not self.workdir:
            QMessageBox.warning(self, "Error", "Work directory is not set.")
            return
        
        try:
            # Import FeatureExtractor here to avoid circular imports
            from app.feature_extractor import FeatureExtractor
            
            # Get the main layout and its widgets
            main_layout = self._layout.itemAt(0).widget()
            right_widget = main_layout.widget(1)  # The placeholder
            
            # Remove the placeholder
            right_widget.setParent(None)
            
            # Create the feature extractor
            self.feature_extractor = FeatureExtractor(self.workdir, self.image_list)
            main_layout.addWidget(self.feature_extractor)
            
            # Set stretch factors again
            main_layout.setStretchFactor(0, 1)  # Left side (image tree)
            main_layout.setStretchFactor(1, 4)  # Right side (feature extractor)
            
            # Populate the tree with camera data
            self.setup_camera_image_tree(self.camera_image_tree)
            
            self.is_initialized = True
            
        except Exception as e:
            error_message = f"Failed to initialize FeatureExtractor: {str(e)}"
            QMessageBox.critical(self, "Error", error_message)
    
    def display_features_for_image(self, item, column):
        """Display features for the selected image"""
        if not self.is_initialized:
            self.initialize()
            
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
            self.setup_basic_ui()
            self.is_initialized = False
            self.initialize()