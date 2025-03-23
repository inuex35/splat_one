# test_images_tab.py

import os
import json
import pytest
from unittest.mock import MagicMock, patch
from PyQt5.QtWidgets import QApplication, QTreeWidgetItem, QDialog
from PyQt5.QtGui import QPixmap
from app.tabs.images_tab import ImagesTab

@pytest.fixture
def mock_camera_model_manager():
    """Mock CameraModelManager for testing"""
    manager = MagicMock()
    manager.get_camera_models.return_value = {
        "Camera1": {
            "projection_type": "perspective",
            "focal_ratio": 1.5
        }
    }
    return manager

@pytest.fixture
def mock_image_processor():
    """Mock ImageProcessor for testing"""
    processor = MagicMock()
    processor.get_sample_image_dimensions.return_value = (1920, 1080)
    processor.resize_images.return_value = 2  # Number of images processed
    processor.restore_original_images.return_value = True
    return processor

@pytest.fixture
def setup_images_tab(mock_qapplication, setup_image_folders):
    """Set up ImagesTab for testing"""
    tab = ImagesTab(setup_image_folders)
    return tab

@patch('app.tabs.images_tab.CameraModelManager')
@patch('app.tabs.images_tab.ImageProcessor')
def test_images_tab_initialize(mock_image_processor_class, mock_model_manager_class, setup_images_tab):
    """Test ImagesTab initialization"""
    # Setup mock returns
    mock_model_manager = MagicMock()
    mock_model_manager_class.return_value = mock_model_manager
    
    mock_processor = MagicMock()
    mock_image_processor_class.return_value = mock_processor
    
    # Initialize the tab
    setup_images_tab.initialize()
    
    # Check that managers were created
    assert mock_model_manager_class.call_count == 1
    assert mock_image_processor_class.call_count == 1
    
    # Check that UI elements were created
    assert setup_images_tab.camera_image_tree is not None
    assert setup_images_tab.image_viewer is not None
    assert setup_images_tab.exif_table is not None
    assert setup_images_tab.is_initialized is True

@patch('app.tabs.images_tab.QPixmap')
@patch('os.path.exists')
def test_display_image_and_exif(mock_exists, mock_pixmap, setup_images_tab, monkeypatch):
    """Test display_image_and_exif method"""
    # Setup mocks
    mock_exists.return_value = True
    mock_pixmap_instance = MagicMock()
    mock_pixmap.return_value = mock_pixmap_instance
    
    # Mock open and json.load
    mock_open = MagicMock()
    mock_json_load = MagicMock(return_value={"camera": "Camera1", "width": 1920})
    
    monkeypatch.setattr('builtins.open', mock_open)
    monkeypatch.setattr('json.load', mock_json_load)
    
    # Create mock item
    item = MagicMock()
    item.childCount.return_value = 0
    item.parent.return_value = MagicMock()  # Not None
    item.text.return_value = "test_image.jpg"
    
    # Initialize necessary UI components
    setup_images_tab.image_viewer = MagicMock()
    setup_images_tab.exif_table = MagicMock()
    setup_images_tab.display_exif_data = MagicMock()  # Mock the method to avoid testing it here
    
    # Call the method
    setup_images_tab.display_image_and_exif(item, 0)
    
    # Check that image was displayed
    assert setup_images_tab.image_viewer.setPixmap.call_count == 1
    assert mock_pixmap.call_count == 1
    
    # Check that EXIF was loaded
    mock_open.assert_called_once()
    mock_json_load.assert_called_once()
    setup_images_tab.display_exif_data.assert_called_once()

@patch('json.dumps')
def test_display_exif_data(mock_json_dumps, setup_images_tab, mock_camera_model_manager):
    """Test display_exif_data method"""
    # Setup
    setup_images_tab.exif_table = MagicMock()
    setup_images_tab.camera_model_manager = mock_camera_model_manager
    mock_json_dumps.return_value = '{"lat": 35.123, "lon": 139.456}'
    
    # Sample EXIF data
    exif_data = {
        "camera": "Camera1",
        "make": "Test Make",
        "model": "Test Model",
        "width": 1920,
        "height": 1080,
        "gps": {"lat": 35.123, "lon": 139.456}
    }
    
    # Call the method
    setup_images_tab.display_exif_data(exif_data)
    
    # Check that table was populated
    assert setup_images_tab.exif_table.setRowCount.call_count >= 1
    assert setup_images_tab.exif_table.insertRow.call_count > 0
    assert setup_images_tab.exif_table.setItem.call_count > 0
    
    # Check that overrides were applied (should be using focal_ratio from mock_camera_model_manager)
    mock_camera_model_manager.get_camera_models.assert_called_once()

@patch('app.tabs.images_tab.QMessageBox')
def test_open_camera_model_editor(mock_msgbox, setup_images_tab, mock_camera_model_manager):
    """Test open_camera_model_editor method"""
    # Setup
    setup_images_tab.camera_model_manager = mock_camera_model_manager
    
    # Call the method
    setup_images_tab.open_camera_model_editor()
    
    # Check that editor was opened
    mock_camera_model_manager.open_camera_model_editor.assert_called_once_with(parent=setup_images_tab)
    
    # Test error case
    setup_images_tab.camera_model_manager = None
    setup_images_tab.open_camera_model_editor()
    mock_msgbox.warning.assert_called_once()

@patch('app.tabs.images_tab.ResolutionDialog')
@patch('app.tabs.images_tab.ExifExtractProgressDialog')
@patch('app.tabs.images_tab.QMessageBox')
def test_resize_images_in_folder(mock_msgbox, mock_progress_dialog, mock_resolution_dialog, 
                              setup_images_tab, mock_image_processor):
    """Test resize_images_in_folder method"""
    # Setup
    setup_images_tab.image_processor = mock_image_processor
    
    # Mock dialog
    mock_dialog = MagicMock()
    mock_dialog.exec_.return_value = QDialog.Accepted
    mock_dialog.get_values.return_value = ("Percentage (%)", 50)
    mock_resolution_dialog.return_value = mock_dialog
    
    # Mock progress dialog
    mock_progress = MagicMock()
    mock_progress_dialog.return_value = mock_progress
    
    # Call the method
    setup_images_tab.resize_images_in_folder()
    
    # Check that dialog was shown and resize was called
    mock_resolution_dialog.assert_called_once_with(1920, 1080, parent=setup_images_tab)
    mock_image_processor.resize_images.assert_called_once_with("Percentage (%)", 50)
    mock_msgbox.information.assert_called_once()
    
    # Test dialog rejected
    mock_dialog.exec_.return_value = QDialog.Rejected
    setup_images_tab.resize_images_in_folder()
    assert mock_image_processor.resize_images.call_count == 1  # Still just one call
    
    # Test error case
    setup_images_tab.image_processor = None
    setup_images_tab.resize_images_in_folder()
    assert mock_msgbox.warning.call_count == 1

@patch('app.tabs.images_tab.QMessageBox')
def test_restore_original_images(mock_msgbox, setup_images_tab, mock_image_processor):
    """Test restore_original_images method"""
    # Setup
    setup_images_tab.image_processor = mock_image_processor
    
    # Call the method
    setup_images_tab.restore_original_images()
    
    # Check that restore was called
    mock_image_processor.restore_original_images.assert_called_once()
    mock_msgbox.information.assert_called_once()
    
    # Test error case
    mock_image_processor.restore_original_images.return_value = False
    setup_images_tab.restore_original_images()
    assert mock_msgbox.warning.call_count == 1
    
    # Test no image processor
    setup_images_tab.image_processor = None
    setup_images_tab.restore_original_images()
    assert mock_msgbox.warning.call_count == 2
