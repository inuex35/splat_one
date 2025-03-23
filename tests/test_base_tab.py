# test_base_tab.py

import pytest
from unittest.mock import MagicMock, patch
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from app.base_tab import BaseTab

class TestTab(BaseTab):
    """Test implementation of BaseTab for testing"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_called = False
        self.refresh_called = False
        self.tab_activated_called = False
        self.tab_deactivated_called = False
    
    def get_tab_name(self):
        return "Test Tab"
    
    def initialize(self):
        self.initialize_called = True
        self.is_initialized = True
    
    def refresh(self):
        self.refresh_called = True
    
    def on_tab_activated(self):
        super().on_tab_activated()
        self.tab_activated_called = True
    
    def on_tab_deactivated(self):
        super().on_tab_deactivated()
        self.tab_deactivated_called = True


def test_base_tab_init():
    """Test BaseTab initialization"""
    workdir = "/test/workdir"
    image_list = ["image1.jpg", "image2.jpg"]
    parent = MagicMock()
    
    tab = TestTab(workdir, image_list, parent)
    
    assert tab.workdir == workdir
    assert tab.image_list == image_list
    assert tab.parent_app == parent
    assert isinstance(tab._layout, QVBoxLayout)
    assert tab.is_initialized is False


def test_base_tab_get_tab_name():
    """Test get_tab_name method"""
    tab = TestTab()
    assert tab.get_tab_name() == "Test Tab"


def test_base_tab_on_tab_activated():
    """Test on_tab_activated method"""
    tab = TestTab()
    tab.on_tab_activated()
    
    assert tab.initialize_called is True
    assert tab.is_initialized is True
    assert tab.tab_activated_called is True


def test_base_tab_update_workdir():
    """Test update_workdir method"""
    tab = TestTab("/old/workdir")
    tab.is_initialized = True
    
    tab.update_workdir("/new/workdir")
    
    assert tab.workdir == "/new/workdir"
    assert tab.refresh_called is True


def test_base_tab_update_image_list():
    """Test update_image_list method"""
    tab = TestTab(image_list=["old.jpg"])
    tab.is_initialized = True
    
    new_images = ["new1.jpg", "new2.jpg"]
    tab.update_image_list(new_images)
    
    assert tab.image_list == new_images
    assert tab.refresh_called is True


@patch('app.base_tab.QTreeWidget')
def test_base_tab_setup_camera_image_tree(mock_tree_widget):
    """Test setup_camera_image_tree method"""
    # This test is simplified due to complexity of mocking file operations
    # A more comprehensive test would use a fixture with actual files
    tab = TestTab("/test/workdir", ["image1.jpg"])
    
    tree = MagicMock()
    callback = MagicMock()
    
    # Call the method
    tab.setup_camera_image_tree(tree, callback)
    
    # Check that tree was cleared
    assert tree.clear.call_count == 1
    
    # If callback provided, it should be connected
    if callback:
        assert tree.itemClicked.connect.call_count == 1