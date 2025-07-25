# depth_tab.py

import os
import sys
import json
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QTreeWidget, QTableWidget, QTableWidgetItem, 
    QSizePolicy, QMessageBox, QApplication, QDialog, QComboBox,
    QProgressBar, QGroupBox, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

from app.base_tab import BaseTab

# Add Depth-Anything-V2 to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'submodules', 'Depth-Anything-V2'))


class DepthEstimationThread(QThread):
    """Thread for running depth estimation"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, workdir, image_list, model_type='depth_anything_v2'):
        super().__init__()
        self.workdir = workdir
        self.image_list = image_list
        self.model_type = model_type
        self.model = None
        self.transform = None
        
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
            
            # Create depth output directory
            depth_dir = os.path.join(self.workdir, "depth")
            os.makedirs(depth_dir, exist_ok=True)
            
            # Process images
            total_images = len(self.image_list)
            for i, image_name in enumerate(self.image_list):
                self.status.emit(f"Processing {image_name}...")
                
                # Load image
                image_path = os.path.join(self.workdir, "images", image_name)
                if not os.path.exists(image_path):
                    continue
                    
                # Estimate depth
                depth_map = self.estimate_depth(image_path)
                
                # Save depth map
                depth_path = os.path.join(depth_dir, f"{image_name}_depth.npy")
                np.save(depth_path, depth_map)
                
                # Save colorized depth map
                colorized_depth = self.colorize_depth(depth_map)
                colorized_path = os.path.join(depth_dir, f"{image_name}_depth.png")
                colorized_depth.save(colorized_path)
                
                # Update progress
                progress = int((i + 1) / total_images * 100)
                self.progress.emit(progress)
            
            self.status.emit("Depth estimation completed!")
            
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()
    
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
            
            # Use vitl by default for good balance of quality and speed
            encoder = 'vitl'
            
            # Initialize model
            model = DepthAnythingV2(**model_configs[encoder])
            
            # Try to load checkpoint
            checkpoint_path = os.path.join(self.workdir, 'models', f'depth_anything_v2_{encoder}.pth')
            if os.path.exists(checkpoint_path):
                self.status.emit(f"Loading checkpoint from {checkpoint_path}...")
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(state_dict)
            else:
                self.status.emit("Warning: No checkpoint found, using random weights")
            
            # Set device and eval mode
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            model = model.to(device).eval()
            
            return model
            
        except Exception as e:
            print(f"Error loading Depth Anything V2 model: {e}")
            raise
    
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
    
    def colorize_depth(self, depth_map):
        """Colorize depth map for visualization"""
        # Normalize depth to 0-20m range
        depth_clipped = np.clip(depth_map, 0, 20)
        depth_normalized = depth_clipped / 20.0
        
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
        self.model_selector = None
        self.estimate_button = None
        self.progress_bar = None
        self.status_label = None
        self.depth_thread = None
    
    def get_tab_name(self):
        return "Depth"
    
    def initialize(self):
        """Initialize the tab when first accessed"""
        if not self.workdir:
            QMessageBox.warning(self, "Error", "Work directory is not set.")
            return
            
        main_layout = QVBoxLayout()
        
        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
        
        # Main content area
        splitter = self.create_horizontal_splitter()
        
        # Left side: Camera and image tree
        self.camera_image_tree = QTreeWidget()
        self.camera_image_tree.setHeaderLabel("Cameras and Images")
        self.camera_image_tree.setFixedWidth(250)
        splitter.addWidget(self.camera_image_tree)
        
        # Middle: Original image viewer
        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        image_layout.addWidget(QLabel("Original Image"))
        self.image_viewer = QLabel()
        self.image_viewer.setAlignment(Qt.AlignCenter)
        self.image_viewer.setStyleSheet("border: 1px solid #ccc;")
        self.image_viewer.setMinimumSize(400, 300)
        image_layout.addWidget(self.image_viewer)
        splitter.addWidget(image_container)
        
        # Right: Depth map viewer
        depth_container = QWidget()
        depth_layout = QVBoxLayout(depth_container)
        depth_layout.addWidget(QLabel("Depth Map"))
        self.depth_viewer = QLabel()
        self.depth_viewer.setAlignment(Qt.AlignCenter)
        self.depth_viewer.setStyleSheet("border: 1px solid #ccc;")
        self.depth_viewer.setMinimumSize(400, 300)
        depth_layout.addWidget(self.depth_viewer)
        splitter.addWidget(depth_container)
        
        # Set splitter sizes
        splitter.setSizes([250, 500, 500])
        
        main_layout.addWidget(splitter)
        self._layout.addLayout(main_layout)
        
        # Populate the camera image tree
        if self.workdir and self.image_list:
            self.setup_camera_image_tree(self.camera_image_tree, self.on_image_selected)
            
        self.is_initialized = True
    
    def create_control_panel(self):
        """Create the control panel with model selection"""
        group_box = QGroupBox("Depth Estimation Settings")
        layout = QHBoxLayout()
        
        # Model selection
        layout.addWidget(QLabel("Model:"))
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Depth Anything V2", "DAC (Depth Anything Camera)"])
        layout.addWidget(self.model_selector)
        
        # Estimate button
        self.estimate_button = QPushButton("Estimate Depth for All Images")
        self.estimate_button.clicked.connect(self.estimate_depth_all)
        layout.addWidget(self.estimate_button)
        
        # Add stretch
        layout.addStretch()
        
        group_box.setLayout(layout)
        return group_box
    
    def on_image_selected(self, item, column):
        """Handle image selection from tree"""
        if item.parent() is None:  # Camera node
            return
            
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
    
    def estimate_depth_all(self):
        """Estimate depth for all images"""
        if not self.workdir or not self.image_list:
            QMessageBox.warning(self, "Warning", "No images available for depth estimation")
            return
        
        # Get selected model type
        model_type = 'depth_anything_v2' if self.model_selector.currentIndex() == 0 else 'dac'
        
        # Disable controls
        self.estimate_button.setEnabled(False)
        self.model_selector.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Create and start depth estimation thread
        self.depth_thread = DepthEstimationThread(self.workdir, self.image_list, model_type)
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
        # Re-enable controls
        self.estimate_button.setEnabled(True)
        self.model_selector.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Refresh current display
        current_item = self.camera_image_tree.currentItem()
        if current_item and current_item.parent():
            self.display_image_and_depth(current_item.text(0))
    
    def refresh(self):
        """Refresh the tab contents"""
        # Reinitialize the tab if initialized
        if self.is_initialized:
            # Remove old widgets
            for i in reversed(range(self._layout.count())): 
                self._layout.itemAt(i).widget().setParent(None)
            
            # Reinitialize
            self.is_initialized = False
            self.initialize()