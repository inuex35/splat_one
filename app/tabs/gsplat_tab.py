# gsplat_tab.py

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QLabel, QTreeWidget, QMessageBox
)
from PyQt5.QtCore import Qt

from app.base_tab import BaseTab

class GsplatTab(BaseTab):
    """Gsplat tab implementation"""
    def __init__(self, workdir=None, image_list=None, parent=None):
        super().__init__(workdir, image_list, parent)
        self.gsplat_manager = None
        self.gsplat_image_tree = None
    
    def get_tab_name(self):
        return "Gsplat"
    
    def initialize(self):
        """Initialize the Gsplat tab"""
        if not self.workdir:
            QMessageBox.warning(self, "Error", "Work directory is not set.")
            return
        
        layout = self.create_horizontal_splitter()
        
        # Left side: Tree of images
        self.gsplat_image_tree = QTreeWidget()
        self.gsplat_image_tree.setHeaderLabel("Images")
        self.gsplat_image_tree.setFixedWidth(250)
        layout.addWidget(self.gsplat_image_tree)
        
        try:
            # Import GsplatManager here to avoid circular imports
            from app.gsplat_manager import GsplatManager
            
            # Right side: GsplatManager widget
            self.gsplat_manager = GsplatManager(self.workdir)
            layout.addWidget(self.gsplat_manager)
            
            # Set stretch factors
            layout.setStretchFactor(0, 1)  # Left side (image tree)
            layout.setStretchFactor(1, 4)  # Right side (gsplat manager)
            
            # Set layout for gsplat tab
            self._layout.addWidget(layout)
            
            # Populate the tree with camera data
            self.setup_camera_image_tree(self.gsplat_image_tree)
            
            # Connect double-click signal to handler
            self.gsplat_image_tree.itemDoubleClicked.connect(self.handle_image_double_click)
            
            self.is_initialized = True
            
        except Exception as e:
            error_message = f"Failed to initialize GsplatManager: {str(e)}"
            QMessageBox.critical(self, "Error", error_message)
            placeholder = QLabel("GsplatManager could not be initialized. See error message.")
            placeholder.setAlignment(Qt.AlignCenter)
            layout.addWidget(placeholder)
            self._layout.addWidget(layout)
    
    def handle_image_double_click(self, item, column):
        """Handle double-click event on image tree item"""
        if item.childCount() == 0 and item.parent() is not None:
            image_name = item.text(0)
            if self.gsplat_manager and hasattr(self.gsplat_manager, 'on_camera_image_tree_double_click'):
                self.gsplat_manager.on_camera_image_tree_double_click(image_name)
    
    def refresh(self):
        """Refresh the tab content"""
        if self.is_initialized:
            # Remove old widgets
            for i in reversed(range(self._layout.count())): 
                self._layout.itemAt(i).widget().setParent(None)
            
            # Reinitialize
            self.is_initialized = False
            self.initialize()