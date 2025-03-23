# Common test fixtures and configurations
import os
import sys
import pytest
from unittest.mock import MagicMock
import tempfile
import shutil

# Add the parent directory to path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def mock_qapplication(monkeypatch):
    """Mock QApplication to avoid GUI during tests"""
    mock_app = MagicMock()
    monkeypatch.setattr('PyQt5.QtWidgets.QApplication', MagicMock(return_value=mock_app))
    return mock_app

@pytest.fixture
def temp_workdir():
    """Create a temporary working directory for tests"""
    temp_dir = tempfile.mkdtemp()
    
    # Create necessary subdirectories
    os.makedirs(os.path.join(temp_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'exif'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'masks'), exist_ok=True)
    
    try:
        yield temp_dir
    finally:
        # Cleanup after tests
        shutil.rmtree(temp_dir)

@pytest.fixture
def sample_image_data():
    """Sample image data for testing"""
    return [
        {'name': 'image1.jpg', 'width': 1920, 'height': 1080, 'camera': 'Camera1'},
        {'name': 'image2.jpg', 'width': 1920, 'height': 1080, 'camera': 'Camera1'},
        {'name': 'image3.jpg', 'width': 3840, 'height': 2160, 'camera': 'Camera2'}
    ]
