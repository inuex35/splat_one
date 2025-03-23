# base_tab.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSplitter, QTreeWidget, QTreeWidgetItem
from PyQt5.QtCore import Qt

class BaseTab(QWidget):
    """Base class for application tabs"""
    def __init__(self, workdir=None, image_list=None, parent=None):
        super().__init__(parent)
        self.workdir = workdir
        self.image_list = image_list or []
        self.parent_app = parent
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.is_initialized = False
    
    def get_tab_name(self):
        """Get the name of the tab. To be overridden by child classes."""
        return "Unnamed Tab"
    
    def initialize(self):
        """Initialize the tab. Must be implemented by child classes."""
        raise NotImplementedError("This method should be implemented by child classes.")
    
    def setup_camera_image_tree(self, tree_widget, select_callback=None):
        """Set up a camera-grouped image tree widget"""
        tree_widget.clear()
        camera_groups = {}
        
        if not self.workdir or not self.image_list:
            return
        
        import json
        import os
        exif_dir = os.path.join(self.workdir, "exif")
        
        for image_name in self.image_list:
            exif_file = os.path.join(exif_dir, image_name + '.exif')
            if os.path.exists(exif_file):
                with open(exif_file, 'r') as f:
                    exif_data = json.load(f)
                camera = exif_data.get('camera', 'Unknown Camera')
            else:
                camera = 'Unknown Camera'
            
            if camera not in camera_groups:
                camera_groups[camera] = []
            
            camera_groups[camera].append(image_name)
        
        for camera, images in camera_groups.items():
            camera_item = QTreeWidgetItem(tree_widget)
            camera_item.setText(0, camera)
            for img in images:
                img_item = QTreeWidgetItem(camera_item)
                img_item.setText(0, img)
        
        # Connect callback if provided
        if select_callback:
            tree_widget.itemClicked.connect(select_callback)
    
    def create_horizontal_splitter(self):
        """Create and return a horizontal splitter"""
        return QSplitter(Qt.Horizontal)
    
    def update_workdir(self, workdir):
        """Update the working directory"""
        self.workdir = workdir
        if self.is_initialized:
            self.refresh()
    
    def update_image_list(self, image_list):
        """Update the image list"""
        self.image_list = image_list
        if self.is_initialized:
            self.refresh()
    
    def refresh(self):
        """Refresh the tab contents. Can be overridden by child classes."""
        pass
    
    def on_tab_activated(self):
        """Called when the tab is activated. Can be overridden by child classes."""
        if not self.is_initialized:
            self.initialize()
            self.is_initialized = True
    
    def on_tab_deactivated(self):
        """Called when the tab is deactivated. Can be overridden by child classes."""
        pass