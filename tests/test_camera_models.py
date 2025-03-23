# test_camera_models.py

import os
import json
import pytest
from unittest.mock import MagicMock, patch
from PyQt5.QtWidgets import QTableWidgetItem, QComboBox
from app.camera_models import CameraModelManager, CameraModelEditor

@pytest.fixture
def sample_camera_models():
    """Sample camera models for testing"""
    return {
        "Camera1": {
            "projection_type": "perspective",
            "width": 1920,
            "height": 1080,
            "focal_ratio": 1.2
        },
        "Camera2": {
            "projection_type": "spherical",
            "width": 3840,
            "height": 2160,
            "focal_ratio": 1.0
        }
    }

@pytest.fixture
def setup_workdir(temp_workdir, sample_camera_models):
    """Set up a temporary working directory with camera model files"""
    # Create camera_models.json
    camera_models_path = os.path.join(temp_workdir, "camera_models.json")
    with open(camera_models_path, "w") as f:
        json.dump(sample_camera_models, f)
    
    # Create camera_models_overrides.json with some overrides
    overrides = {
        "Camera1": {
            "focal_ratio": 1.5  # Override just one parameter
        }
    }
    overrides_path = os.path.join(temp_workdir, "camera_models_overrides.json")
    with open(overrides_path, "w") as f:
        json.dump(overrides, f)
    
    return temp_workdir


def test_camera_model_manager_init(setup_workdir, sample_camera_models):
    """Test CameraModelManager initialization and loading"""
    manager = CameraModelManager(setup_workdir)
    
    # Check that models are loaded
    assert manager.camera_models is not None
    
    # Check that Camera1's focal_ratio was overridden
    assert manager.camera_models["Camera1"]["focal_ratio"] == 1.5
    
    # Check that other parameters are preserved
    assert manager.camera_models["Camera1"]["projection_type"] == "perspective"
    assert manager.camera_models["Camera2"]["projection_type"] == "spherical"


def test_camera_model_manager_get_models(setup_workdir):
    """Test get_camera_models method"""
    manager = CameraModelManager(setup_workdir)
    models = manager.get_camera_models()
    
    assert "Camera1" in models
    assert "Camera2" in models
    assert models["Camera1"]["focal_ratio"] == 1.5  # Overridden value


@patch('app.camera_models.CameraModelEditor')
def test_camera_model_manager_open_editor(mock_editor, setup_workdir):
    """Test open_camera_model_editor method"""
    # Set up the mock to return True from exec_()
    editor_instance = MagicMock()
    editor_instance.exec_.return_value = True
    mock_editor.return_value = editor_instance
    
    manager = CameraModelManager(setup_workdir)
    parent = MagicMock()
    
    # Call the method
    result = manager.open_camera_model_editor(parent)
    
    # Check that the editor was created with correct parameters
    mock_editor.assert_called_once_with(manager.camera_models, setup_workdir, parent=parent)
    assert result is True
    
    # Now test with no workdir
    manager.workdir = None
    result = manager.open_camera_model_editor(parent)
    assert result is False
    # QMessageBox warning should be called, but we can't test that directly here


@patch('app.camera_models.QTableWidgetItem')
@patch('app.camera_models.QComboBox')
def test_camera_model_editor_load_models(mock_combo, mock_item, setup_workdir, sample_camera_models):
    """Test CameraModelEditor.load_camera_models"""
    # Create mock for QTableWidgetItem
    mock_item_instance = MagicMock()
    mock_item.return_value = mock_item_instance
    
    # Create mock for QComboBox
    mock_combo_instance = MagicMock()
    mock_combo.return_value = mock_combo_instance
    
    # Create mock for the table
    mock_table = MagicMock()
    
    # Create the editor with mocked table
    editor = CameraModelEditor(sample_camera_models, setup_workdir)
    editor.table = mock_table
    
    # Call the method
    editor.load_camera_models()
    
    # Check that rows were added to the table
    assert mock_table.setRowCount.call_count >= 1
    assert mock_table.insertRow.call_count > 0
    assert mock_table.setItem.call_count > 0
    
    # Check that combo boxes were added for projection_type
    assert mock_combo.call_count > 0


@patch('app.camera_models.json.dump')
@patch('app.camera_models.QMessageBox')
def test_camera_model_editor_save_changes(mock_msgbox, mock_json_dump, setup_workdir, sample_camera_models):
    """Test CameraModelEditor.save_changes"""
    editor = CameraModelEditor(sample_camera_models, setup_workdir)
    
    # Create mock table with data
    editor.table = MagicMock()
    editor.table.rowCount.return_value = 3
    
    # Set up the table to return different types of cells
    def mock_get_item(row, col):
        # For the first row
        if row == 0:
            if col == 0:  # Key column
                item = MagicMock(spec=QTableWidgetItem)
                item.text.return_value = "Camera1"
                return item
            elif col == 1:  # Parameter column
                item = MagicMock(spec=QTableWidgetItem)
                item.text.return_value = "focal_ratio"
                return item
            elif col == 2:  # Value column
                item = MagicMock(spec=QTableWidgetItem)
                item.text.return_value = "1.8"  # Float value
                return item
        # For the second row
        elif row == 1:
            if col == 0:
                item = MagicMock(spec=QTableWidgetItem)
                item.text.return_value = "Camera1"
                return item
            elif col == 1:
                item = MagicMock(spec=QTableWidgetItem)
                item.text.return_value = "width"
                return item
            elif col == 2:
                item = MagicMock(spec=QTableWidgetItem)
                item.text.return_value = "1920"  # Integer value
                return item
        # For the third row with ComboBox
        elif row == 2:
            if col == 0:
                item = MagicMock(spec=QTableWidgetItem)
                item.text.return_value = "Camera1"
                return item
            elif col == 1:
                item = MagicMock(spec=QTableWidgetItem)
                item.text.return_value = "projection_type"
                return item
            elif col == 2:
                # Return None for cell with widget
                return None
        return None
    
    editor.table.item.side_effect = mock_get_item
    
    # Mock cellWidget for the third row, returning a combo box
    def mock_cell_widget(row, col):
        if row == 2 and col == 2:
            combo = MagicMock(spec=QComboBox)
            combo.currentText.return_value = "spherical"
            return combo
        return None
    
    editor.table.cellWidget.side_effect = mock_cell_widget
    
    # Call the method
    editor.save_changes()
    
    # Check that json.dump was called with the expected values
    assert mock_json_dump.call_count == 1
    
    # Get the arguments passed to json.dump
    args, kwargs = mock_json_dump.call_args
    updated_models = args[0]
    
    # Check the updated models
    assert "Camera1" in updated_models
    assert updated_models["Camera1"]["focal_ratio"] == 1.8
    assert updated_models["Camera1"]["width"] == 1920
    assert updated_models["Camera1"]["projection_type"] == "spherical"
    
    # Check that success message was shown
    assert mock_msgbox.information.call_count == 1
