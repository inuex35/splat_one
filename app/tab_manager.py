# tab_manager.py

from PyQt5.QtWidgets import QTabWidget
from app.base_tab import BaseTab

class TabManager(QTabWidget):
    """Manager for application tabs"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tab_instances = {}
        self.parent_app = parent
        self.currentChanged.connect(self.on_tab_changed)
        self.active_tab_index = -1
    
    def register_tab(self, tab_class, tab_name=None, *args, **kwargs):
        """Register a tab with the manager"""
        # Create the tab instance
        tab_instance = tab_class(*args, parent=self.parent_app, **kwargs)
        
        # Use provided name or get from the tab
        if tab_name is None and isinstance(tab_instance, BaseTab):
            tab_name = tab_instance.get_tab_name()
        elif tab_name is None:
            tab_name = tab_class.__name__
        
        # Add to tab widget
        index = self.addTab(tab_instance, tab_name)
        
        # Store reference
        self.tab_instances[index] = tab_instance
        
        # If this is the first tab, set it as active
        if index == 0:
            self.active_tab_index = 0
            # Initialize the first tab immediately
            if isinstance(tab_instance, BaseTab) and hasattr(tab_instance, 'on_tab_activated'):
                tab_instance.on_tab_activated()
        
        return index
    
    def get_tab_instance(self, index):
        """Get the tab instance at the given index"""
        return self.tab_instances.get(index)
    
    def get_current_tab(self):
        """Get the currently active tab instance"""
        index = self.currentIndex()
        return self.get_tab_instance(index)
    
    def on_tab_changed(self, index):
        """Handle tab changed event"""
        # Deactivate previous tab
        prev_tab = self.get_tab_instance(self.active_tab_index)
        if prev_tab and hasattr(prev_tab, 'on_tab_deactivated'):
            prev_tab.on_tab_deactivated()
        
        # Activate new tab
        current_tab = self.get_tab_instance(index)
        if current_tab and hasattr(current_tab, 'on_tab_activated'):
            current_tab.on_tab_activated()
        
        # Update active index
        self.active_tab_index = index
    
    def update_all_tabs(self, workdir=None, image_list=None):
        """Update all tabs with new workdir and/or image list"""
        for tab_instance in self.tab_instances.values():
            if hasattr(tab_instance, 'update_workdir') and workdir is not None:
                tab_instance.update_workdir(workdir)
            
            if hasattr(tab_instance, 'update_image_list') and image_list is not None:
                tab_instance.update_image_list(image_list)