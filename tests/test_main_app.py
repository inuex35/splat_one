# test_main_app.py

import os
import sys
import pytest
import shutil
from unittest.mock import MagicMock, patch
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox, QFileDialog
from app.main_app import MainApp, VideoProcessDialog

@pytest.fixture
def setup_main_app(mock_qapplication):
    """Set up MainApp for testing with mocked dependencies"""
    with patch('app.main_app.TabManager'), \
         patch('app.main_app.QTimer'), \
         patch('app.main_app.QToolBar'):
        app = MainApp()
        app.show_start_dialog = MagicMock()  # Prevent actual dialog on init
        return app


def test_main_app_init(setup_main_app):
    """Test MainApp initialization"""
    app = setup_main_app
    
    # Check attributes
    assert app.workdir is None
    assert app.image_list == []
    assert app.tab_manager is not None
    assert app.camera_model_manager is None
    assert app.image_processor is None
    
    # Check that QTimer was set to call show_start_dialog
    from app.main_app import QTimer
    assert QTimer.singleShot.call_count == 1


def test_setup_ui(setup_main_app):
    """Test setup_ui method"""
    app = setup_main_app
    
    # Reset mocks
    app.tab_manager.reset_mock()
    
    # Call method again
    app.setup_ui()
    
    # Check that TabManager was set as central widget
    assert app.centralWidget() == app.tab_manager


def test_register_tabs(setup_main_app):
    """Test register_tabs method"""
    app = setup_main_app
    
    # Reset mock
    app.tab_manager.reset_mock()
    
    # Call method
    app.register_tabs()
    
    # Check that register_tab was called
    assert app.tab_manager.register_tab.call_count >= 1


@patch('app.main_app.QMessageBox.question')
def test_show_start_dialog_video(mock_question, setup_main_app):
    """Test show_start_dialog selecting video processing"""
    app = setup_main_app
    app.process_video = MagicMock()
    app.select_image_folder = MagicMock()
    
    # Choose video processing
    mock_question.return_value = QMessageBox.Yes
    
    # Call method
    app.show_start_dialog()
    
    # Check that process_video was called
    app.process_video.assert_called_once()
    assert app.select_image_folder.call_count == 0


@patch('app.main_app.QMessageBox.question')
def test_show_start_dialog_image_folder(mock_question, setup_main_app):
    """Test show_start_dialog selecting image folder"""
    app = setup_main_app
    app.process_video = MagicMock()
    app.select_image_folder = MagicMock()
    
    # Choose image folder
    mock_question.return_value = QMessageBox.No
    
    # Call method
    app.show_start_dialog()
    
    # Check that select_image_folder was called
    app.select_image_folder.assert_called_once()
    assert app.process_video.call_count == 0


@patch('app.main_app.QFileDialog.getExistingDirectory')
@patch('os.makedirs')
@patch('os.listdir')
@patch('os.path.exists')
@patch('shutil.copy')
def test_select_image_folder(mock_copy, mock_exists, mock_listdir, mock_makedirs, 
                             mock_get_dir, setup_main_app):
    """Test select_image_folder method"""
    app = setup_main_app
    app.load_workdir = MagicMock()
    
    # Set up mocks
    test_dir = "/test/workdir"
    mock_get_dir.return_value = test_dir
    mock_listdir.return_value = ["image1.jpg", "image2.jpg", "file.txt"]
    mock_exists.return_value = False  # Images don't exist in target dir
    
    # Call method
    app.select_image_folder()
    
    # Check that workdir was set
    assert app.workdir == test_dir
    
    # Check that images directory was created
    mock_makedirs.assert_called_with(os.path.join(test_dir, "images"), exist_ok=True)
    
    # Check that images were copied
    assert mock_copy.call_count == 2  # Two image files
    
    # Check that load_workdir was called
    app.load_workdir.assert_called_once()
    
    # Test cancel case
    mock_get_dir.return_value = ""
    with patch('app.main_app.QMessageBox.warning') as mock_warning, \
         patch('app.main_app.sys.exit') as mock_exit:
        app.select_image_folder()
        mock_warning.assert_called_once()
        mock_exit.assert_called_once_with(1)


@patch('app.main_app.QFileDialog.getOpenFileName')
@patch('app.main_app.VideoProcessDialog')
@patch('app.main_app.ExifExtractProgressDialog')
@patch('app.main_app.VideoProcessCommand')
@patch('app.main_app.ImageProcessor')
def test_process_video(mock_image_processor, mock_video_command, mock_progress_dialog,
                      mock_dialog, mock_get_file, setup_main_app):
    """Test process_video method"""
    app = setup_main_app
    app.load_workdir = MagicMock()
    
    # Set up mocks
    test_file = "/test/video.mp4"
    mock_get_file.return_value = (test_file, "")
    
    dialog_instance = MagicMock()
    dialog_instance.exec_.return_value = QDialog.Accepted
    dialog_instance.get_values.return_value = {
        "import_path": "/test/import_path",
        "method": "Interval",
        "interval": 0.5,
        "distance": -1,
        "geotag_source": "video",
        "geotag_source_path": None,
        "offset_time": 0,
        "use_gpx": True
    }
    mock_dialog.return_value = dialog_instance
    
    progress_instance = MagicMock()
    mock_progress_dialog.return_value = progress_instance
    
    video_command_instance = MagicMock()
    mock_video_command.return_value = video_command_instance
    
    processor_instance = MagicMock()
    mock_image_processor.return_value = processor_instance
    
    # Mock file existence checks
    with patch('os.makedirs') as mock_makedirs, \
         patch('os.path.exists') as mock_exists, \
         patch('app.main_app.QApplication.processEvents') as mock_process_events:
        
        mock_exists.return_value = True
        
        # Call method
        app.process_video()
        
        # Check that dialog was shown
        mock_dialog.assert_called_once()
        dialog_instance.exec_.assert_called_once()
        
        # Check that progress dialog was shown
        mock_progress_dialog.assert_called_once()
        progress_instance.show.assert_called_once()
        
        # Check that VideoProcessCommand was used
        mock_video_command.assert_called_once()
        video_command_instance.run.assert_called_once()
        
        # Check that ImageProcessor was used
        mock_image_processor.assert_called_once()
        processor_instance.apply_exif_from_mapillary_json.assert_called_once()
        
        # Check that workdir was set and load_workdir called
        assert app.workdir == "/test/import_path"
        app.load_workdir.assert_called_once()
    
    # Test cancel case
    dialog_instance.exec_.return_value = QDialog.Rejected
    with patch('app.main_app.QMessageBox.warning') as mock_warning, \
         patch('app.main_app.sys.exit') as mock_exit:
        app.process_video()
        mock_warning.assert_called_once()
        mock_exit.assert_called_once_with(1)


@patch('os.path.exists')
@patch('os.listdir')
def test_load_workdir_with_exif(mock_listdir, mock_exists, setup_main_app):
    """Test load_workdir method when EXIF data exists"""
    app = setup_main_app
    app.workdir = "/test/workdir"
    
    # Mock file operations
    mock_exists.return_value = True  # All paths exist
    mock_listdir.return_value = ["image1.jpg", "image2.jpg"]
    
    # Mock camera model manager
    with patch('app.main_app.CameraModelManager') as mock_manager, \
         patch('app.main_app.ImageProcessor') as mock_processor:
        
        manager_instance = MagicMock()
        mock_manager.return_value = manager_instance
        
        processor_instance = MagicMock()
        mock_processor.return_value = processor_instance
        
        # Call method
        app.load_workdir()
        
        # Check that image list was populated
        assert app.image_list == ["image1.jpg", "image2.jpg"]
        
        # Check that managers were created
        mock_manager.assert_called_once_with(app.workdir)
        mock_processor.assert_called_once_with(app.workdir)
        
        # Check that tab manager was updated
        if app.tab_manager:
            app.tab_manager.update_all_tabs.assert_called_once_with(
                workdir=app.workdir, image_list=app.image_list
            )


@patch('os.path.exists')
@patch('os.listdir')
def test_load_workdir_without_exif(mock_listdir, mock_exists, setup_main_app):
    """Test load_workdir method when EXIF data doesn't exist"""
    app = setup_main_app
    app.workdir = "/test/workdir"
    
    # Mock file operations to simulate missing EXIF
    def mock_exists_side_effect(path):
        return "images" in path  # Only images directory exists
    
    mock_exists.side_effect = mock_exists_side_effect
    mock_listdir.return_value = ["image1.jpg", "image2.jpg"]
    
    # Mock progress dialog
    with patch('app.main_app.ExifExtractProgressDialog') as mock_progress, \
         patch('app.main_app.CameraModelManager') as mock_manager, \
         patch('app.main_app.ImageProcessor') as mock_processor, \
         patch('app.main_app.QApplication.processEvents') as mock_process, \
         patch('shutil.copy') as mock_copy, \
         patch('app.main_app.DataSet') as mock_dataset, \
         patch('app.main_app.extract_metadata') as mock_extract:
        
        progress_instance = MagicMock()
        mock_progress.return_value = progress_instance
        
        manager_instance = MagicMock()
        mock_manager.return_value = manager_instance
        
        processor_instance = MagicMock()
        mock_processor.return_value = processor_instance
        
        dataset_instance = MagicMock()
        mock_dataset.return_value = dataset_instance
        
        # Call method
        app.load_workdir()
        
        # Check that progress dialog was shown
        mock_progress.assert_called_once()
        progress_instance.show.assert_called_once()
        
        # Check that metadata extraction was called
        mock_copy.assert_called_once()  # Config file copy
        mock_dataset.assert_called_once_with(app.workdir)
        mock_extract.run_dataset.assert_called_once_with(dataset_instance)
        
        # Check that camera model editor was opened
        manager_instance.open_camera_model_editor.assert_called_once_with(parent=app)


def test_video_process_dialog(mock_qapplication):
    """Test VideoProcessDialog"""
    dialog = VideoProcessDialog()
    
    # Test toggle_sampling_inputs
    dialog.sampling_method_combo = MagicMock()
    dialog.distance_input = MagicMock()
    dialog.interval_input = MagicMock()
    
    # Test Interval method
    dialog.sampling_method_combo.currentText.return_value = "Interval"
    dialog.toggle_sampling_inputs()
    dialog.distance_input.setDisabled.assert_called_with(True)
    dialog.interval_input.setDisabled.assert_called_with(False)
    
    # Test Distance method
    dialog.sampling_method_combo.currentText.return_value = "Distance"
    dialog.toggle_sampling_inputs()
    dialog.distance_input.setDisabled.assert_called_with(False)
    dialog.interval_input.setDisabled.assert_called_with(True)
    
    # Test get_sampling_values
    dialog.distance_input.text.return_value = "5"
    dialog.interval_input.text.return_value = "0.5"
    
    # With Distance method
    dialog.sampling_method_combo.currentText.return_value = "Distance"
    method, interval, distance = dialog.get_sampling_values()
    assert method == "Distance"
    assert interval == -1
    assert distance == 5.0
    
    # With Interval method
    dialog.sampling_method_combo.currentText.return_value = "Interval"
    method, interval, distance = dialog.get_sampling_values()
    assert method == "Interval"
    assert interval == 0.5
    assert distance == -1
    
    # Test get_values
    dialog.import_path_input = MagicMock()
    dialog.import_path_input.text.return_value = "/test/import"
    
    dialog.geotag_source_combo = MagicMock()
    dialog.geotag_source_combo.currentText.return_value = "video"
    
    dialog.geotag_source_path_input = MagicMock()
    dialog.geotag_source_path_input.text.return_value = ""
    
    dialog.interpolation_offset_input = MagicMock()
    dialog.interpolation_offset_input.text.return_value = "1.5"
    
    dialog.interpolation_use_gpx_checkbox = MagicMock()
    dialog.interpolation_use_gpx_checkbox.isChecked.return_value = True
    
    # Mock get_sampling_values
    dialog.get_sampling_values = MagicMock(return_value=("Interval", 0.5, -1))
    
    values = dialog.get_values()
    assert values["import_path"] == "/test/import"
    assert values["method"] == "Interval"
    assert values["interval"] == 0.5
    assert values["distance"] == -1
    assert values["geotag_source"] == "video"
    assert values["geotag_source_path"] is None
    assert values["offset_time"] == 1.5
    assert values["use_gpx"] is True