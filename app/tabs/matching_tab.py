# matching_tab.py

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QLabel, QTreeWidget, QMessageBox
)
from PyQt5.QtCore import Qt

from app.base_tab import BaseTab

class MatchingTab(BaseTab):
    """Matching tab implementation"""
    def __init__(self, workdir=None, image_list=None, parent=None):
        super().__init__(workdir, image_list, parent)
        self.matching_viewer = None
        self.camera_image_tree_left = None
        self.camera_image_tree_right = None
    
    def get_tab_name(self):
        return "Matching"
    
    def initialize(self):
        """Initialize the Matching tab"""
        if not self.workdir:
            QMessageBox.warning(self, "Error", "Work directory is not set.")
            return
        
        layout = self.create_horizontal_splitter()
        
        # Left side: Tree of images (Column 1)
        self.camera_image_tree_left = QTreeWidget()
        self.camera_image_tree_left.setHeaderLabel("Cameras and Images - Left")
        self.camera_image_tree_left.setFixedWidth(250)
        layout.addWidget(self.camera_image_tree_left)
        
        try:
            # Import FeatureMatching here to avoid circular imports
            from app.feature_matching import FeatureMatching
            
            # Center: FeatureMatching widget
            self.matching_viewer = FeatureMatching(workdir=self.workdir, image_list=self.image_list)
            layout.addWidget(self.matching_viewer)
            
            # Right side: Tree of images (Column 2)
            self.camera_image_tree_right = QTreeWidget()
            self.camera_image_tree_right.setHeaderLabel("Cameras and Images - Right")
            self.camera_image_tree_right.setFixedWidth(250)
            layout.addWidget(self.camera_image_tree_right)
            
            # Set layout for matching tab
            self._layout.addWidget(layout)
            
            # Populate each tree with camera data
            self.setup_camera_image_tree(self.camera_image_tree_left, self.on_image_selected_left)
            self.setup_camera_image_tree(self.camera_image_tree_right, self.on_image_selected_right)
            
            # Set stretch factors
            layout.setStretchFactor(0, 1)  # Left tree
            layout.setStretchFactor(1, 3)  # Matching viewer
            layout.setStretchFactor(2, 1)  # Right tree
            
            self.is_initialized = True
            
        except Exception as e:
            error_message = f"Failed to initialize FeatureMatching: {str(e)}"
            QMessageBox.critical(self, "Error", error_message)
            placeholder = QLabel("FeatureMatching could not be initialized. See error message.")
            placeholder.setAlignment(Qt.AlignCenter)
            layout.addWidget(placeholder)
            self._layout.addWidget(layout)
    
    def on_image_selected_left(self, item, column):
        """Handle image selection in the left camera tree"""
        if item.childCount() == 0 and item.parent() is not None:
            image_name = item.text(0)
            if self.matching_viewer:
                self.matching_viewer.load_image_by_name(image_name, position="left")
    
    def on_image_selected_right(self, item, column):
        """Handle image selection in the right camera tree"""
        if item.childCount() == 0 and item.parent() is not None:
            image_name = item.text(0)
            if self.matching_viewer:
                self.matching_viewer.load_image_by_name(image_name, position="right")
    
    def refresh(self):
        """Refresh the tab content"""
        if self.is_initialized:
            # Remove old widgets
            for i in reversed(range(self._layout.count())): 
                self._layout.itemAt(i).widget().setParent(None)
            
            # Reinitialize
            self.is_initialized = False
            self.initialize()