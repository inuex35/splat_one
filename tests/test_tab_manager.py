# test_tab_manager.py

import pytest
from unittest.mock import MagicMock, patch
from PyQt5.QtWidgets import QWidget
from app.tab_manager import TabManager
from app.base_tab import BaseTab

class MockTab(BaseTab):
    """Mock implementation of BaseTab for testing"""
    def __init__(self, workdir=None, image_list=None, parent=None, custom_arg=None):
        super().__init__(workdir, image_list, parent)
        self.custom_arg = custom_arg
        self.activated = False
        self.deactivated = False
    
    def get_tab_name(self):
        return "Mock Tab"
    
    def initialize(self):
        self.is_initialized = True
    
    def on_tab_activated(self):
        super().on_tab_activated()
        self.activated = True
    
    def on_tab_deactivated(self):
        super().on_tab_deactivated()
        self.deactivated = True


class SimpleWidget(QWidget):
    """Simple widget for testing non-BaseTab tabs"""
    def __init__(self, parent=None):
        super().__init__(parent)


@pytest.fixture
def tab_manager(mock_qapplication):
    """Create a TabManager for testing"""
    return TabManager()


def test_tab_manager_init(tab_manager):
    """Test TabManager initialization"""
    assert isinstance(tab_manager, TabManager)
    assert tab_manager.tab_instances == {}
    assert tab_manager.active_tab_index == -1


def test_register_tab_with_base_tab(tab_manager):
    """Test registering a BaseTab-derived tab"""
    # Register a tab
    index = tab_manager.register_tab(MockTab, workdir="/test/workdir", custom_arg="test")
    
    # Check that the tab was added
    assert index == 0
    assert index in tab_manager.tab_instances
    assert isinstance(tab_manager.tab_instances[index], MockTab)
    assert tab_manager.tab_instances[index].workdir == "/test/workdir"
    assert tab_manager.tab_instances[index].custom_arg == "test"
    assert tab_manager.tabText(index) == "Mock Tab"


def test_register_tab_with_widget(tab_manager):
    """Test registering a non-BaseTab widget"""
    # Register a widget with explicit name
    index = tab_manager.register_tab(SimpleWidget, tab_name="Simple Widget")
    
    # Check that the tab was added
    assert index == 0
    assert index in tab_manager.tab_instances
    assert isinstance(tab_manager.tab_instances[index], SimpleWidget)
    assert tab_manager.tabText(index) == "Simple Widget"
    
    # Register a widget without explicit name
    index = tab_manager.register_tab(SimpleWidget)
    
    # Check that the tab was added with class name
    assert index == 1
    assert tab_manager.tabText(index) == "SimpleWidget"


def test_get_tab_instance(tab_manager):
    """Test get_tab_instance method"""
    # Register tabs
    index1 = tab_manager.register_tab(MockTab)
    index2 = tab_manager.register_tab(SimpleWidget)
    
    # Get instances
    tab1 = tab_manager.get_tab_instance(index1)
    tab2 = tab_manager.get_tab_instance(index2)
    
    # Check correct instances
    assert isinstance(tab1, MockTab)
    assert isinstance(tab2, SimpleWidget)
    
    # Test invalid index
    assert tab_manager.get_tab_instance(999) is None


@patch('app.tab_manager.TabManager.currentIndex')
def test_get_current_tab(mock_current_index, tab_manager):
    """Test get_current_tab method"""
    # Register tabs
    index1 = tab_manager.register_tab(MockTab)
    index2 = tab_manager.register_tab(SimpleWidget)
    
    # Mock currentIndex
    mock_current_index.return_value = index1
    
    # Get current tab
    current_tab = tab_manager.get_current_tab()
    
    # Check correct instance
    assert isinstance(current_tab, MockTab)
    
    # Change current tab
    mock_current_index.return_value = index2
    current_tab = tab_manager.get_current_tab()
    assert isinstance(current_tab, SimpleWidget)


def test_on_tab_changed(tab_manager):
    """Test on_tab_changed method"""
    # Register tabs
    index1 = tab_manager.register_tab(MockTab)
    index2 = tab_manager.register_tab(MockTab)
    
    # Get tab instances
    tab1 = tab_manager.get_tab_instance(index1)
    tab2 = tab_manager.get_tab_instance(index2)
    
    # Initially, no tab is active
    assert tab_manager.active_tab_index == -1
    
    # Activate first tab
    tab_manager.on_tab_changed(index1)
    assert tab1.activated is True
    assert tab1.deactivated is False
    assert tab2.activated is False
    assert tab2.deactivated is False
    assert tab_manager.active_tab_index == index1
    
    # Change to second tab
    tab_manager.on_tab_changed(index2)
    assert tab1.activated is True
    assert tab1.deactivated is True
    assert tab2.activated is True
    assert tab2.deactivated is False
    assert tab_manager.active_tab_index == index2


def test_update_all_tabs(tab_manager):
    """Test update_all_tabs method"""
    # Register tabs
    index1 = tab_manager.register_tab(MockTab, workdir="/old/workdir", image_list=["old.jpg"])
    index2 = tab_manager.register_tab(SimpleWidget)  # Non-BaseTab widget
    
    # Get first tab instance
    tab1 = tab_manager.get_tab_instance(index1)
    
    # Mock update methods
    tab1.update_workdir = MagicMock()
    tab1.update_image_list = MagicMock()
    
    # Call update_all_tabs
    new_workdir = "/new/workdir"
    new_image_list = ["new1.jpg", "new2.jpg"]
    tab_manager.update_all_tabs(workdir=new_workdir, image_list=new_image_list)
    
    # Check that update methods were called
    tab1.update_workdir.assert_called_once_with(new_workdir)
    tab1.update_image_list.assert_called_once_with(new_image_list)
    
    # Test with only workdir
    tab1.update_workdir.reset_mock()
    tab1.update_image_list.reset_mock()
    tab_manager.update_all_tabs(workdir="/another/workdir")
    tab1.update_workdir.assert_called_once_with("/another/workdir")
    assert tab1.update_image_list.call_count == 0
    
    # Test with only image_list
    tab1.update_workdir.reset_mock()
    tab1.update_image_list.reset_mock()
    tab_manager.update_all_tabs(image_list=["another.jpg"])
    assert tab1.update_workdir.call_count == 0
    tab1.update_image_list.assert_called_once_with(["another.jpg"])
