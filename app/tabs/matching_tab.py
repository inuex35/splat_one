# matching_tab.py

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QTreeWidget, QMessageBox
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
        
        # Set up basic UI structure
        self.setup_basic_ui()
    
    def get_tab_name(self):
        return "Matching"
    
    def setup_basic_ui(self):
        """Set up the basic UI structure"""
        # Create a main horizontal splitter
        main_layout = self.create_horizontal_splitter()
        
        # Left side: Combined tree widget container
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create label for left and right image selection
        left_layout.addWidget(QLabel("Left Image Selection:"))
        
        # Tree for left image
        self.camera_image_tree_left = QTreeWidget()
        self.camera_image_tree_left.setHeaderLabel("Cameras and Images - Left")
        left_layout.addWidget(self.camera_image_tree_left)
        
        # Add a separator label
        left_layout.addWidget(QLabel("Right Image Selection:"))
        
        # Tree for right image
        self.camera_image_tree_right = QTreeWidget()
        self.camera_image_tree_right.setHeaderLabel("Cameras and Images - Right")
        left_layout.addWidget(self.camera_image_tree_right)
        
        # Add the left widget to the main layout with fixed width
        left_widget.setFixedWidth(300)  # A bit wider to accommodate the trees
        main_layout.addWidget(left_widget)
        
        # Right side: Placeholder for matching viewer
        right_widget = QLabel("Matching Viewer will be displayed here.")
        right_widget.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(right_widget)
        
        # Set stretch factors
        main_layout.setStretchFactor(0, 1)  # Left side (trees)
        main_layout.setStretchFactor(1, 3)  # Right side (matching viewer)
        
        # Set layout for matching tab
        self._layout.addWidget(main_layout)
    
    def initialize(self):
        """Initialize the Matching tab with data"""
        if not self.workdir:
            QMessageBox.warning(self, "Error", "Work directory is not set.")
            return
        
        try:
            # Import FeatureMatching here to avoid circular imports
            from app.feature_matching import FeatureMatching
            
            # Get the main layout and its widgets
            main_layout = self._layout.itemAt(0).widget()
            right_widget = main_layout.widget(1)  # The placeholder
            
            # Remove the placeholder
            right_widget.setParent(None)
            
            # Create the matching viewer
            self.matching_viewer = FeatureMatching(workdir=self.workdir, image_list=self.image_list)
            main_layout.addWidget(self.matching_viewer)
            
            # Set stretch factors again (they might be reset after widget changes)
            main_layout.setStretchFactor(0, 1)  # Left side (trees)
            main_layout.setStretchFactor(1, 3)  # Right side (matching viewer)
            
            # Populate the trees with camera data
            self.setup_camera_image_tree(self.camera_image_tree_left, self.on_image_selected_left)
            self.setup_camera_image_tree(self.camera_image_tree_right, self.on_image_selected_right)
            
            self.is_initialized = True
            
        except Exception as e:
            error_message = f"Failed to initialize FeatureMatching: {str(e)}"
            QMessageBox.critical(self, "Error", error_message)
    
    def on_image_selected_left(self, item, column):
        """Handle image selection in the left camera tree"""
        if not self.is_initialized:
            self.initialize()
            
        if item.childCount() == 0 and item.parent() is not None:
            image_name = item.text(0)
            if self.matching_viewer:
                self.matching_viewer.load_image_by_name(image_name, position="left")
    
    def on_image_selected_right(self, item, column):
        """Handle image selection in the right camera tree"""
        if not self.is_initialized:
            self.initialize()
            
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
            self.setup_basic_ui()
            self.is_initialized = False
            self.initialize()