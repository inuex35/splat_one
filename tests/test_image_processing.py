# test_image_processing.py

import os
import json
import pytest
import shutil
from unittest.mock import MagicMock, patch
from PIL import Image
from app.image_processing import ImageProcessor, ResolutionDialog, ExifExtractProgressDialog

@pytest.fixture
def setup_image_folders(temp_workdir):
    """Set up image folders in temporary directory"""
    # Create necessary directories
    images_dir = os.path.join(temp_workdir, "images")
    exif_dir = os.path.join(temp_workdir, "exif")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(exif_dir, exist_ok=True)
    
    # Create a test image
    test_img_path = os.path.join(images_dir, "test_image.jpg")
    img = Image.new('RGB', (100, 50), color = 'red')
    img.save(test_img_path)
    
    # Create another test image
    test_img2_path = os.path.join(images_dir, "test_image2.jpg")
    img2 = Image.new('RGB', (200, 150), color = 'blue')
    img2.save(test_img2_path)
    
    return temp_workdir


def test_image_processor_init(setup_image_folders):
    """Test ImageProcessor initialization"""
    processor = ImageProcessor(setup_image_folders)
    
    assert processor.workdir == setup_image_folders
    assert os.path.exists(processor.images_folder)
    assert processor.images_folder == os.path.join(setup_image_folders, "images")
    assert processor.exif_folder == os.path.join(setup_image_folders, "exif")


def test_get_sample_image_dimensions(setup_image_folders):
    """Test get_sample_image_dimensions method"""
    processor = ImageProcessor(setup_image_folders)
    width, height = processor.get_sample_image_dimensions()
    
    # Could be either of the test images, so check both possibilities
    assert (width == 100 and height == 50) or (width == 200 and height == 150)
    
    # Test with empty folder
    images_dir = processor.images_folder
    for file in os.listdir(images_dir):
        if file.endswith('.jpg'):
            os.remove(os.path.join(images_dir, file))
    
    # Now no images should be found
    width, height = processor.get_sample_image_dimensions()
    assert width is None and height is None


def test_resize_images(setup_image_folders):
    """Test resize_images method"""
    processor = ImageProcessor(setup_image_folders)
    
    # Get original dimensions of test image
    test_img_path = os.path.join(processor.images_folder, "test_image.jpg")
    with Image.open(test_img_path) as img:
        original_width = img.width
        original_height = img.height
    
    # Resize to 50%
    mock_callback = MagicMock()
    num_processed = processor.resize_images("Percentage (%)", 50, mock_callback)
    
    # Check that the progress callback was called
    assert mock_callback.call_count > 0
    
    # Check that the image was resized
    with Image.open(test_img_path) as img:
        assert img.width == original_width // 2
        assert img.height == original_height // 2
    
    # Check that backup folder was created
    assert os.path.exists(processor.images_org_folder)
    
    # Test width-based resizing
    processor.resize_images("Width (px)", 75)
    with Image.open(test_img_path) as img:
        assert img.width == 75
    
    # Test height-based resizing
    processor.resize_images("Height (px)", 30)
    with Image.open(test_img_path) as img:
        assert img.height == 30


def test_restore_original_images(setup_image_folders):
    """Test restore_original_images method"""
    processor = ImageProcessor(setup_image_folders)
    
    # First resize to create a backup
    processor.resize_images("Percentage (%)", 50)
    
    # Get current dimensions after resize
    test_img_path = os.path.join(processor.images_folder, "test_image.jpg")
    with Image.open(test_img_path) as img:
        resized_width = img.width
        resized_height = img.height
    
    # Now restore
    result = processor.restore_original_images()
    assert result is True
    
    # Check that the image was restored to original dimensions
    with Image.open(test_img_path) as img:
        assert img.width != resized_width  # Should not be the resized dimensions
        assert img.height != resized_height
    
    # Test when no backup exists
    shutil.rmtree(processor.images_org_folder)
    result = processor.restore_original_images()
    assert result is False


@patch('app.image_processing.piexif.load')
@patch('app.image_processing.piexif.dump')
@patch('app.image_processing.fractions.Fraction')
@patch('app.image_processing.Image.open')
def test_apply_exif_from_mapillary_json(mock_image_open, mock_fraction, mock_dump, mock_load, setup_image_folders, tmp_path):
    """Test apply_exif_from_mapillary_json method"""
    # Set up mock Image
    mock_img = MagicMock()
    mock_img.info = {}
    mock_image_open.return_value.__enter__.return_value = mock_img
    
    # Set up mock fractions.Fraction
    mock_fraction.return_value.numerator = 1
    mock_fraction.return_value.denominator = 2
    
    # Set up mock load
    mock_load.side_effect = KeyError  # Force new exif_dict creation
    
    # Set up mock dump
    mock_dump.return_value = b'fake exif data'
    
    # Create a test JSON file with Mapillary data
    json_path = os.path.join(tmp_path, "mapillary_image_description.json")
    images_dir = os.path.join(tmp_path, "video_frames")
    os.makedirs(images_dir, exist_ok=True)
    
    # Create a test image
    test_img_path = os.path.join(images_dir, "frame_0001.jpg")
    with open(test_img_path, 'w') as f:
        f.write("fake image data")
    
    mapillary_data = [
        {
            "filename": "frame_0001.jpg",
            "MAPLatitude": 35.6895,
            "MAPLongitude": 139.6917,
            "MAPAltitude": 10.5,
            "MAPCaptureTime": "2023_01_01_12_30_45_000",
            "MAPOrientation": 1
        }
    ]
    
    with open(json_path, 'w') as f:
        json.dump(mapillary_data, f)
    
    # Create the processor and call the method
    processor = ImageProcessor(setup_image_folders)
    processed_count = processor.apply_exif_from_mapillary_json(json_path, images_dir)
    
    # Check results
    assert processed_count == 1
    assert mock_image_open.call_count == 1
    assert mock_dump.call_count == 1
    assert not os.path.exists(images_dir)  # Should be renamed
    assert os.path.exists(os.path.join(tmp_path, "images"))  # New folder name


@patch('app.image_processing.QDialogButtonBox')
def test_resolution_dialog(mock_dialog_button_box, mock_qapplication):
    """Test ResolutionDialog"""
    dialog = ResolutionDialog(1920, 1080)
    
    # Test initial values
    assert dialog.original_width == 1920
    assert dialog.original_height == 1080
    assert dialog.aspect_ratio == 1920 / 1080
    
    # Test update_label method
    dialog.resize_method_combo = MagicMock()
    dialog.value_input = MagicMock()
    
    # Test Percentage method
    dialog.resize_method_combo.currentText.return_value = "Percentage (%)"
    dialog.update_label()
    dialog.value_input.setText.assert_called_with("100")
    
    # Test Width method
    dialog.resize_method_combo.currentText.return_value = "Width (px)"
    dialog.update_label()
    dialog.value_input.setText.assert_called_with("1920")
    
    # Test Height method
    dialog.resize_method_combo.currentText.return_value = "Height (px)"
    dialog.update_label()
    dialog.value_input.setText.assert_called_with("1080")
    
    # Test get_values method
    dialog.resize_method_combo.currentText.return_value = "Percentage (%)"
    dialog.value_input.text.return_value = "75"
    method, value = dialog.get_values()
    assert method == "Percentage (%)"
    assert value == 75.0


def test_exif_extract_progress_dialog(mock_qapplication):
    """Test ExifExtractProgressDialog"""
    # Just basic instantiation test
    dialog = ExifExtractProgressDialog("Processing...")
    assert dialog.windowTitle() == "Extracting EXIF Data"
    
    # Test with default message
    dialog = ExifExtractProgressDialog()
    assert dialog.isModal() is True
