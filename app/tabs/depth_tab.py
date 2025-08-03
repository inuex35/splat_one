# depth_tab.py

import os
import sys
import json
import numpy as np
import torch
import requests
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QTreeWidget, QTableWidget, QTableWidgetItem, 
    QSizePolicy, QMessageBox, QApplication, QDialog, QComboBox,
    QProgressBar, QGroupBox, QRadioButton, QButtonGroup, QTabWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

from app.base_tab import BaseTab
from app.point_cloud_viewer import PointCloudViewer

# Add Depth-Anything-V2 to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'submodules', 'Depth-Anything-V2'))


class WeightDownloadThread(QThread):
    """Thread for downloading model weights"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, workdir, model_type='vitl'):
        super().__init__()
        self.workdir = workdir
        self.model_type = model_type
        self._stop_flag = False
        
        # Model download URLs
        self.download_urls = {
            'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true',
            'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true',
            'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true'
        }
        
    def run(self):
        try:
            # Create models directory
            models_dir = os.path.join(self.workdir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            # Get download URL
            if self.model_type not in self.download_urls:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            url = self.download_urls[self.model_type]
            filename = f'depth_anything_v2_{self.model_type}.pth'
            filepath = os.path.join(models_dir, filename)
            
            # Check if file already exists
            if os.path.exists(filepath):
                self.status.emit(f"Weight file {filename} already exists!")
                self.finished.emit()
                return
            
            self.status.emit(f"Downloading {filename}...")
            
            # Download with progress tracking
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        # Check stop flag
                        if self._stop_flag:
                            self.status.emit("Download stopped")
                            return
                        
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = int((downloaded_size / total_size) * 100)
                            self.progress.emit(progress)
            
            if not self._stop_flag:
                self.status.emit(f"Download completed: {filename}")
            
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()
    
    def stop(self):
        """Stop the download"""
        self._stop_flag = True


class DepthEstimationThread(QThread):
    """Thread for running depth estimation"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, workdir, image_list, model_type='depth_anything_v2', model_size='vitl'):
        super().__init__()
        self.workdir = workdir
        self.image_list = image_list
        self.model_type = model_type
        self.model_size = model_size
        self.model = None
        self.transform = None
        self._stop_flag = False
        
    def run(self):
        try:
            # Load model
            self.status.emit(f"Loading {self.model_type} model...")
            if self.model_type == 'depth_anything_v2':
                self.model = self.load_depth_anything_v2()
            elif self.model_type == 'dac':
                self.model = self.load_dac_model()
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Check if stopped during model loading
            if self._stop_flag:
                self.status.emit("Depth estimation stopped")
                return
            
            # Create depth output directory
            depth_dir = os.path.join(self.workdir, "depth")
            os.makedirs(depth_dir, exist_ok=True)
            
            # Process images
            total_images = len(self.image_list)
            for i, image_name in enumerate(self.image_list):
                # Check stop flag
                if self._stop_flag:
                    self.status.emit("Depth estimation stopped")
                    return
                
                self.status.emit(f"Processing {image_name}...")
                
                # Load image
                image_path = os.path.join(self.workdir, "images", image_name)
                if not os.path.exists(image_path):
                    continue
                    
                # Estimate depth (already includes post-processing)
                depth_map = self.estimate_depth(image_path)
                
                # Check stop flag again after estimation
                if self._stop_flag:
                    self.status.emit("Depth estimation stopped")
                    return
                
                # Save processed depth map
                depth_path = os.path.join(depth_dir, f"{image_name}_depth.npy")
                np.save(depth_path, depth_map)
                
                # Save colorized depth map
                colorized_depth = self.colorize_depth(depth_map)
                colorized_path = os.path.join(depth_dir, f"{image_name}_depth.png")
                colorized_depth.save(colorized_path)
                
                # Update progress
                progress = int((i + 1) / total_images * 100)
                self.progress.emit(progress)
            
            if not self._stop_flag:
                self.status.emit("Depth estimation completed!")
            
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()
    
    def stop(self):
        """Stop the depth estimation"""
        self._stop_flag = True
    
    def load_depth_anything_v2(self):
        """Load Depth Anything V2 model"""
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
            
            self.status.emit("Loading Depth Anything V2 model...")
            
            # Model configurations
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            
            # Use selected model size
            encoder = self.model_size
            
            # Initialize model
            model = DepthAnythingV2(**model_configs[encoder])
            
            # Try to load checkpoint
            checkpoint_path = os.path.join(self.workdir, 'models', f'depth_anything_v2_{encoder}.pth')
            if os.path.exists(checkpoint_path):
                self.status.emit(f"Loading checkpoint from {checkpoint_path}...")
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(state_dict)
            else:
                # Auto-download the weight file
                self.status.emit(f"Checkpoint not found. Auto-downloading {encoder} weights...")
                success = self.auto_download_weights(encoder)
                if success:
                    self.status.emit(f"Loading downloaded checkpoint from {checkpoint_path}...")
                    state_dict = torch.load(checkpoint_path, map_location='cpu')
                    model.load_state_dict(state_dict)
                else:
                    self.status.emit("Warning: Failed to download weights, using random weights")
            
            # Set device and eval mode
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            model = model.to(device).eval()
            
            return model
            
        except Exception as e:
            print(f"Error loading Depth Anything V2 model: {e}")
            raise
    
    def auto_download_weights(self, model_type):
        """Auto-download weights for the specified model type"""
        try:
            # Model download URLs
            download_urls = {
                'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true',
                'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true',
                'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true'
            }
            
            if model_type not in download_urls:
                return False
            
            # Create models directory
            models_dir = os.path.join(self.workdir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            url = download_urls[model_type]
            filename = f'depth_anything_v2_{model_type}.pth'
            filepath = os.path.join(models_dir, filename)
            
            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return True
            
        except Exception as e:
            print(f"Error auto-downloading weights: {e}")
            return False
    
    def load_dac_model(self):
        """Load model for camera-aware depth estimation"""
        # For now, use vits model for faster inference
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
            
            self.status.emit("Loading Depth Anything V2 (small) model for camera-aware estimation...")
            
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}
            }
            
            encoder = 'vits'  # Use smaller model for camera-aware mode
            
            model = DepthAnythingV2(**model_configs[encoder])
            
            checkpoint_path = os.path.join(self.workdir, 'models', f'depth_anything_v2_{encoder}.pth')
            if os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(state_dict)
            else:
                # Auto-download the weight file
                self.status.emit(f"Checkpoint not found. Auto-downloading {encoder} weights...")
                success = self.auto_download_weights(encoder)
                if success:
                    self.status.emit(f"Loading downloaded checkpoint from {checkpoint_path}...")
                    state_dict = torch.load(checkpoint_path, map_location='cpu')
                    model.load_state_dict(state_dict)
                else:
                    self.status.emit("Warning: Failed to download weights, using random weights")
            
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            model = model.to(device).eval()
            
            return model
            
        except Exception as e:
            print(f"Error loading DAC model: {e}")
            return self.load_depth_anything_v2()
    
    def estimate_depth(self, image_path):
        """Estimate depth for an image"""
        try:
            # Load image with PIL
            image_pil = Image.open(image_path).convert('RGB')
            
            # Convert to numpy array in BGR format (as expected by the model)
            image_np = np.array(image_pil)
            # Convert RGB to BGR by reversing the channel order
            image_bgr = image_np[:, :, ::-1]
            
            # Use the model's built-in infer_image method
            # This method handles all preprocessing internally
            depth_map = self.model.infer_image(image_bgr)
            
            # Post-process the depth map
            depth_map = self.post_process_depth(depth_map)
            
            # The output is already a numpy array in HxW format
            return depth_map
            
        except Exception as e:
            print(f"Error during depth estimation: {e}")
            # Return a dummy depth map
            if 'image_pil' in locals():
                w, h = image_pil.size
                depth_map = np.ones((h, w), dtype=np.float32)
            else:
                depth_map = np.ones((512, 512), dtype=np.float32)
            return depth_map
    
    def post_process_depth(self, depth_map):
        """Post-process depth map for better visualization"""
        # Remove any invalid values
        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get valid depth range
        valid_depths = depth_map[depth_map > 0]
        if len(valid_depths) == 0:
            return depth_map
        
        # Use percentile-based clipping to handle outliers
        min_depth = np.percentile(valid_depths, 1)  # 1st percentile
        max_depth = np.percentile(valid_depths, 99)  # 99th percentile
        
        # Ensure min_depth < max_depth
        if min_depth >= max_depth:
            min_depth = np.min(valid_depths)
            max_depth = np.max(valid_depths)
            if min_depth >= max_depth:
                return depth_map
        
        # Clip depth map to valid range
        depth_map = np.clip(depth_map, min_depth, max_depth)
        
        # Normalize to 0-1 range for better visualization
        depth_map = (depth_map - min_depth) / (max_depth - min_depth)
        
        return depth_map
    
    def colorize_depth(self, depth_map):
        """Colorize depth map for visualization"""
        # Remove any invalid values
        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Print debug info for depth range
        print(f"Depth map stats - min: {np.min(depth_map):.4f}, max: {np.max(depth_map):.4f}, mean: {np.mean(depth_map):.4f}")
        
        # Get valid depth range (exclude outliers)
        valid_depths = depth_map[depth_map > 0]
        if len(valid_depths) == 0:
            # If no valid depths, create a uniform depth map
            depth_normalized = np.ones_like(depth_map) * 0.5
            print("No valid depths found, using uniform depth map")
        else:
            # Use percentile-based normalization to handle outliers
            min_depth = np.percentile(valid_depths, 1)  # 1st percentile
            max_depth = np.percentile(valid_depths, 99)  # 99th percentile
            
            print(f"Valid depth range - 1st percentile: {min_depth:.4f}, 99th percentile: {max_depth:.4f}")
            
            # Ensure min_depth < max_depth
            if min_depth >= max_depth:
                min_depth = np.min(valid_depths)
                max_depth = np.max(valid_depths)
                print(f"Adjusted range - min: {min_depth:.4f}, max: {max_depth:.4f}")
                if min_depth >= max_depth:
                    depth_normalized = np.ones_like(depth_map) * 0.5
                    print("Still no valid range, using uniform depth map")
                else:
                    depth_normalized = (depth_map - min_depth) / (max_depth - min_depth)
            else:
                depth_normalized = (depth_map - min_depth) / (max_depth - min_depth)
            
            # Clip to valid range
            depth_normalized = np.clip(depth_normalized, 0, 1)
        
        # Apply matplotlib's turbo colormap
        colormap = cm.get_cmap('turbo')
        colored = colormap(depth_normalized)
        
        # Convert to RGB (remove alpha channel) and scale to 0-255
        colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        
        # Convert to PIL Image
        colorized_image = Image.fromarray(colored_rgb)
        
        return colorized_image


class DepthTab(BaseTab):
    """Depth estimation tab implementation"""
    def __init__(self, workdir=None, image_list=None, parent=None):
        super().__init__(workdir, image_list, parent)
        self.camera_image_tree = None
        self.image_viewer = None
        self.depth_viewer = None
        self.point_cloud_viewer = None
        self.view_tab_widget = None
        self.model_selector = None
        self.estimate_button = None
        self.download_button = None
        self.progress_bar = None
        self.status_label = None
        self.depth_thread = None
        self.download_thread = None
        self.is_estimating = False
        self.is_downloading = False
    
    def get_tab_name(self):
        return "Depth"
    
    def initialize(self):
        """Initialize the tab when first accessed"""
        if not self.workdir:
            QMessageBox.warning(self, "Error", "Work directory is not set.")
            return
            
        # Set up basic UI structure like features tab
        self.setup_basic_ui()
        
        # Initialize with data
        self.initialize_with_data()
        
        self.is_initialized = True
        
        # Initialize model type change handler
        self.on_model_type_changed()
    
    def setup_basic_ui(self):
        """Set up the basic UI structure like features tab"""
        # Create main layout
        main_layout = QVBoxLayout()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
        
        # Main content area - horizontal splitter like features tab
        layout = self.create_horizontal_splitter()
        
        # Left side: Tree of images grouped by camera
        self.camera_image_tree = QTreeWidget()
        self.camera_image_tree.setHeaderLabel("Cameras and Images")
        self.camera_image_tree.setFixedWidth(250)
        layout.addWidget(self.camera_image_tree)
        
        # Right side: Tab widget for different views
        self.view_tab_widget = QTabWidget()
        
        # Image and Depth tab
        self.depth_viewer_container = QWidget()
        self.depth_viewer_layout = QVBoxLayout(self.depth_viewer_container)
        
        # Add placeholder initially
        placeholder = QLabel("Select an image to view depth estimation")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("border: 1px solid #ccc; color: #666;")
        self.depth_viewer_layout.addWidget(placeholder)
        
        self.view_tab_widget.addTab(self.depth_viewer_container, "Image & Depth")
        
        # Point Cloud tab - lazy initialization
        self.point_cloud_viewer = None
        self.point_cloud_placeholder = QLabel("Initializing point cloud viewer...")
        self.point_cloud_placeholder.setAlignment(Qt.AlignCenter)
        self.point_cloud_placeholder.setStyleSheet("border: 1px solid #ccc; color: #666;")
        self.view_tab_widget.addTab(self.point_cloud_placeholder, "Point Cloud")
        
        # Initialize on tab change
        self.view_tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # Control panel at bottom right
        control_panel = self.create_control_panel()
        
        # Right container
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.addWidget(self.view_tab_widget)
        right_layout.addWidget(control_panel)
        
        layout.addWidget(right_container)
        
        # Set stretch factors like features tab
        layout.setStretchFactor(0, 1)  # Left side (image tree)
        layout.setStretchFactor(1, 4)  # Right side (depth viewer + controls)
        
        main_layout.addWidget(layout)
        self._layout.addLayout(main_layout)
        
        # Connect signals
        self.camera_image_tree.itemClicked.connect(self.on_image_selected)
    
    def initialize_with_data(self):
        """Initialize the depth tab with data"""
        try:
            # Remove the placeholder
            for i in reversed(range(self.depth_viewer_layout.count())):
                widget = self.depth_viewer_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            
            # Create depth viewer widgets
            self.create_depth_viewer_widgets()
            
            # Populate the camera image tree
            if self.workdir and self.image_list:
                self.setup_camera_image_tree(self.camera_image_tree, self.on_image_selected)
            
        except Exception as e:
            error_message = f"Failed to initialize depth tab: {str(e)}"
            QMessageBox.critical(self, "Error", error_message)
    
    def create_depth_viewer_widgets(self):
        """Create depth viewer widgets"""
        # Create a vertical splitter for original image and depth map
        depth_splitter = QSplitter(Qt.Vertical)
        
        # Original image viewer
        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        image_layout.addWidget(QLabel("Original Image"))
        self.image_viewer = QLabel()
        self.image_viewer.setAlignment(Qt.AlignCenter)
        self.image_viewer.setStyleSheet("border: 1px solid #ccc;")
        self.image_viewer.setMinimumSize(400, 300)
        image_layout.addWidget(self.image_viewer)
        depth_splitter.addWidget(image_container)
        
        # Depth map viewer
        depth_container = QWidget()
        depth_layout = QVBoxLayout(depth_container)
        depth_layout.addWidget(QLabel("Depth Map"))
        self.depth_viewer = QLabel()
        self.depth_viewer.setAlignment(Qt.AlignCenter)
        self.depth_viewer.setStyleSheet("border: 1px solid #ccc;")
        self.depth_viewer.setMinimumSize(400, 300)
        depth_layout.addWidget(self.depth_viewer)
        depth_splitter.addWidget(depth_container)
        
        # Set splitter sizes for vertical split
        depth_splitter.setSizes([300, 300])
        
        self.depth_viewer_layout.addWidget(depth_splitter)
    
    def create_control_panel(self):
        """Create the control panel with model selection and download button"""
        group_box = QGroupBox("Depth Estimation Settings")
        layout = QVBoxLayout()
        
        # Model selection section
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model Type:"))
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Depth Anything V2", "DAC (Depth Anything Camera)"])
        self.model_selector.currentIndexChanged.connect(self.on_model_type_changed)
        model_layout.addWidget(self.model_selector)
        layout.addLayout(model_layout)
        
        # Model size selection section
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Model Size:"))
        self.size_selector = QComboBox()
        self.size_selector.addItems(["Small (vits) - 24.8MB - Fast", "Base (vitb) - 97.5MB - Balanced", "Large (vitl) - 335.3MB - Best"])
        self.size_selector.setCurrentIndex(2)  # Default to Large
        size_layout.addWidget(self.size_selector)
        layout.addLayout(size_layout)
        
        # Buttons section
        button_layout = QHBoxLayout()
        
        # Download weights button
        self.download_button = QPushButton("Download Weights")
        self.download_button.clicked.connect(self.download_weights)
        button_layout.addWidget(self.download_button)
        
        # Estimate/Stop button
        self.estimate_button = QPushButton("Estimate Depth for All Images")
        self.estimate_button.clicked.connect(self.toggle_depth_estimation)
        button_layout.addWidget(self.estimate_button)
        
        layout.addLayout(button_layout)
        
        group_box.setLayout(layout)
        return group_box
    
    def download_weights(self):
        """Download model weights"""
        if not self.workdir:
            QMessageBox.warning(self, "Warning", "Work directory is not set.")
            return
        
        # If already downloading, stop the download
        if self.is_downloading:
            if self.download_thread:
                self.download_thread.stop()
            return
        
        # Get selected model type and size
        if self.model_selector.currentIndex() == 0:  # Depth Anything V2
            # Get size from size selector
            size_index = self.size_selector.currentIndex()
            if size_index == 0:
                model_type = 'vits'
            elif size_index == 1:
                model_type = 'vitb'
            else:
                model_type = 'vitl'
        else:  # DAC (Depth Anything Camera)
            model_type = 'vits'  # Use small model for camera-aware mode
        
        # Check if file already exists
        models_dir = os.path.join(self.workdir, 'models')
        filename = f'depth_anything_v2_{model_type}.pth'
        filepath = os.path.join(models_dir, filename)
        
        if os.path.exists(filepath):
            reply = QMessageBox.question(self, "File Exists", 
                                       f"Weight file {filename} already exists. Do you want to download it again?",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
        
        # Start download
        self.is_downloading = True
        self.download_button.setText("Stop Download")
        
        # Disable other controls
        self.estimate_button.setEnabled(False)
        self.model_selector.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Create and start download thread
        self.download_thread = WeightDownloadThread(self.workdir, model_type)
        self.download_thread.progress.connect(self.update_progress)
        self.download_thread.status.connect(self.update_status)
        self.download_thread.error.connect(self.handle_download_error)
        self.download_thread.finished.connect(self.download_finished)
        self.download_thread.start()
    
    def handle_download_error(self, error_message):
        """Handle download error"""
        QMessageBox.critical(self, "Download Error", f"Weight download failed: {error_message}")
        self.status_label.setText("Download failed")
    
    def download_finished(self):
        """Handle completion of weight download"""
        # Reset download state
        self.is_downloading = False
        self.download_button.setText("Download Weights")
        
        # Re-enable controls
        self.download_button.setEnabled(True)
        self.estimate_button.setEnabled(True)
        self.model_selector.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Show success message only if not stopped
        if self.download_thread and not self.download_thread._stop_flag:
            QMessageBox.information(self, "Success", "Weight download completed successfully!")
    
    def on_image_selected(self, item, column):
        """Handle image selection from tree"""
        if not self.is_initialized:
            self.initialize()
            
        if item.childCount() == 0 and item.parent() is not None:
            image_name = item.text(0)
            self.display_image_and_depth(image_name)
    
    def display_image_and_depth(self, image_name):
        """Display original image and its depth map if available"""
        if not self.workdir:
            return
            
        # Display original image
        image_path = os.path.join(self.workdir, "images", image_name)
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(self.image_viewer.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_viewer.setPixmap(scaled_pixmap)
        
        # Display depth map if available
        depth_path = os.path.join(self.workdir, "depth", f"{image_name}_depth.png")
        if os.path.exists(depth_path):
            depth_pixmap = QPixmap(depth_path)
            scaled_depth = depth_pixmap.scaled(self.depth_viewer.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.depth_viewer.setPixmap(scaled_depth)
        else:
            self.depth_viewer.setText("No depth map available")
        
        # Update point cloud if available and initialized
        if self.point_cloud_viewer:
            self.point_cloud_viewer.load_point_cloud_from_images(image_name)
    
    def toggle_depth_estimation(self):
        """Toggle depth estimation (start/stop)"""
        if not self.workdir or not self.image_list:
            QMessageBox.warning(self, "Warning", "No images available for depth estimation")
            return
        
        # If already estimating, stop the estimation
        if self.is_estimating:
            if self.depth_thread:
                self.depth_thread.stop()
            return
        
        # Get selected model type and size
        if self.model_selector.currentIndex() == 0:  # Depth Anything V2
            model_type = 'depth_anything_v2'
            # Get size from size selector
            size_index = self.size_selector.currentIndex()
            if size_index == 0:
                model_size = 'vits'
            elif size_index == 1:
                model_size = 'vitb'
            else:
                model_size = 'vitl'
        else:  # DAC (Depth Anything Camera)
            model_type = 'dac'
            model_size = 'vits'  # DAC only supports small model
        
        # Check if required weight files exist and offer to download if missing
        models_dir = os.path.join(self.workdir, 'models')
        weight_file = os.path.join(models_dir, f'depth_anything_v2_{model_size}.pth')
        
        if not os.path.exists(weight_file):
            reply = QMessageBox.question(self, "Weights Missing", 
                                       f"Required weight file 'depth_anything_v2_{model_size}.pth' not found. Would you like to download it now?",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.download_weights()
                return
            else:
                QMessageBox.information(self, "Info", "Depth estimation will proceed with random weights (poor quality expected)")
        
        # Start estimation
        self.is_estimating = True
        self.estimate_button.setText("Stop Estimation")
        
        # Disable other controls
        self.model_selector.setEnabled(False)
        self.download_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Create and start depth estimation thread
        self.depth_thread = DepthEstimationThread(self.workdir, self.image_list, model_type, model_size)
        self.depth_thread.progress.connect(self.update_progress)
        self.depth_thread.status.connect(self.update_status)
        self.depth_thread.error.connect(self.handle_error)
        self.depth_thread.finished.connect(self.depth_estimation_finished)
        self.depth_thread.start()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def update_status(self, message):
        """Update status message"""
        self.status_label.setText(message)
    
    def handle_error(self, error_message):
        """Handle depth estimation error"""
        QMessageBox.critical(self, "Error", f"Depth estimation failed: {error_message}")
        self.status_label.setText("Error occurred")
    
    def depth_estimation_finished(self):
        """Handle completion of depth estimation"""
        # Reset estimation state
        self.is_estimating = False
        self.estimate_button.setText("Estimate Depth for All Images")
        
        # Re-enable controls
        self.estimate_button.setEnabled(True)
        self.model_selector.setEnabled(True)
        self.download_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Refresh current display
        current_item = self.camera_image_tree.currentItem()
        if current_item and current_item.parent():
            self.display_image_and_depth(current_item.text(0))
            
        # Update point cloud viewer workdir
        if self.point_cloud_viewer:
            self.point_cloud_viewer.set_workdir(self.workdir)
    
    def on_model_type_changed(self):
        """Handle model type change"""
        if self.model_selector.currentIndex() == 0:  # Depth Anything V2
            # Enable all size options
            self.size_selector.setEnabled(True)
            self.size_selector.setCurrentIndex(2)  # Default to Large
        else:  # DAC (Depth Anything Camera)
            # DAC only supports small model
            self.size_selector.setCurrentIndex(0)  # Small
            self.size_selector.setEnabled(False)
    
    def on_tab_changed(self, index):
        """Handle tab change"""
        if index == 1 and self.point_cloud_viewer is None:  # Point cloud tab selected
            try:
                # Initialize point cloud viewer
                self.point_cloud_viewer = PointCloudViewer(self.workdir)
                
                # Remove placeholder and replace with point cloud viewer
                self.view_tab_widget.removeTab(1)
                self.view_tab_widget.insertTab(1, self.point_cloud_viewer, "Point Cloud")
                self.view_tab_widget.setCurrentIndex(1)
                
                # Display point cloud if image is selected
                current_item = self.camera_image_tree.currentItem()
                if current_item and current_item.parent():
                    self.point_cloud_viewer.load_point_cloud_from_images(current_item.text(0))
                    
            except Exception as e:
                print(f"Point cloud viewer initialization error: {e}")
                # Keep placeholder on error
                pass
    
    def refresh(self):
        """Refresh the tab content"""
        if self.is_initialized:
            # Remove old widgets
            for i in reversed(range(self._layout.count())): 
                self._layout.itemAt(i).widget().setParent(None)
            
            # Reinitialize
            self.setup_basic_ui()
            self.is_initialized = False
            self.initialize()
            
            # Update point cloud viewer workdir
            if self.point_cloud_viewer:
                self.point_cloud_viewer.set_workdir(self.workdir)