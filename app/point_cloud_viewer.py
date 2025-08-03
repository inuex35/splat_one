import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QSpinBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QVector3D
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import os

from app.point_cloud_utils import rgb_depth_to_pointcloud, load_image_and_depth, downsample_pointcloud

class PointCloudViewer(QWidget):
    """Point cloud viewer widget (pyqtgraph version)"""
    
    def __init__(self, workdir=None):
        super().__init__()
        self.workdir = workdir
        self.current_points = None
        self.current_colors = None
        self.point_cloud_item = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        
        # pyqtgraph 3D viewer (same method as reconstruction tab)
        self.viewer = gl.GLViewWidget()
        self.viewer.setFocusPolicy(Qt.NoFocus)
        self.viewer.setCameraPosition(distance=10)
        layout.addWidget(self.viewer)
        
        # Control panel
        control_layout = QHBoxLayout()
        
        # Point count adjustment
        control_layout.addWidget(QLabel("Point Count:"))
        self.point_count_spinbox = QSpinBox()
        self.point_count_spinbox.setRange(1000, 100000)
        self.point_count_spinbox.setValue(10000)
        self.point_count_spinbox.valueChanged.connect(self.update_point_cloud)
        control_layout.addWidget(self.point_count_spinbox)
        
        # Point size adjustment
        control_layout.addWidget(QLabel("Point Size:"))
        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setRange(1, 10)
        self.point_size_slider.setValue(3)
        self.point_size_slider.valueChanged.connect(self.update_point_size)
        control_layout.addWidget(self.point_size_slider)
        
        # Reset button
        self.reset_button = QPushButton("Reset Camera")
        self.reset_button.clicked.connect(self.reset_camera)
        control_layout.addWidget(self.reset_button)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
    def load_point_cloud_from_images(self, image_name):
        """Load and display point cloud from images and depth"""
        if not self.workdir:
            return
            
        try:
            # Image and depth paths
            image_path = os.path.join(self.workdir, "images", image_name)
            depth_path = os.path.join(self.workdir, "depth", f"{image_name}_depth.png")
            
            # Load images and depth
            rgb_image, depth_image = load_image_and_depth(image_path, depth_path)
            
            # Generate point cloud
            points, colors = rgb_depth_to_pointcloud(rgb_image, depth_image)
            
            # Save point cloud
            self.current_points = points
            self.current_colors = colors
            
            # Display point cloud
            self.update_point_cloud()
            
        except Exception as e:
            print(f"Point cloud generation error: {e}")
            self.clear_point_cloud()
    
    def update_point_cloud(self):
        """Update and display point cloud"""
        if self.current_points is None or self.current_colors is None:
            return
            
        # Downsample point cloud
        target_points = self.point_count_spinbox.value()
        points, colors = downsample_pointcloud(self.current_points, self.current_colors, target_points)
        
        # Remove existing point cloud
        self.clear_point_cloud()
        
        # Add new point cloud
        if len(points) > 0:
            # Create point cloud item (same method as reconstruction tab)
            self.point_cloud_item = gl.GLScatterPlotItem(
                pos=points,
                color=colors,
                size=self.point_size_slider.value(),
                pxMode=True
            )
            self.viewer.addItem(self.point_cloud_item)
    
    def update_point_size(self):
        """Update point size"""
        if self.point_cloud_item:
            self.point_cloud_item.setData(size=self.point_size_slider.value())
    
    def clear_point_cloud(self):
        """Clear point cloud"""
        if self.point_cloud_item:
            self.viewer.removeItem(self.point_cloud_item)
            self.point_cloud_item = None
    
    def reset_camera(self):
        """Reset camera position"""
        self.viewer.setCameraPosition(distance=10)
    
    def set_workdir(self, workdir):
        """Set working directory"""
        self.workdir = workdir 