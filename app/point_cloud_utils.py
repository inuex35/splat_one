import numpy as np
from PIL import Image
import os

def rgb_depth_to_pointcloud(rgb_image, depth_image, fx=1000, fy=1000, cx=None, cy=None):
    """
    Generate point cloud from RGB image and depth image
    
    Args:
        rgb_image: RGB image (PIL Image or numpy array)
        depth_image: Depth image (PIL Image or numpy array)
        fx, fy: Camera focal length
        cx, cy: Camera optical center (uses image center if None)
    
    Returns:
        points: Point cloud coordinates (N, 3)
        colors: Point cloud colors (N, 3)
    """
    # Convert images to numpy arrays
    if isinstance(rgb_image, Image.Image):
        rgb_array = np.array(rgb_image)
    else:
        rgb_array = rgb_image
    
    if isinstance(depth_image, Image.Image):
        depth_array = np.array(depth_image)
    else:
        depth_array = depth_image
    
    # Convert depth image to RGB if grayscale
    if len(depth_array.shape) == 2:
        depth_array = np.stack([depth_array] * 3, axis=-1)
    
    # Use first channel if depth image is RGB
    if len(depth_array.shape) == 3:
        depth_array = depth_array[:, :, 0]
    
    # Get image dimensions
    height, width = depth_array.shape
    
    # Use image center if optical center is not specified
    if cx is None:
        cx = width / 2
    if cy is None:
        cy = height / 2
    
    # Create mesh grid
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Normalize depth values (0-255 to 0-1)
    depth_normalized = depth_array.astype(np.float32) / 255.0
    
    # Calculate 3D coordinates
    Z = depth_normalized * 10.0  # Adjust depth scale as needed
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    
    # Extract only points with valid depth values
    valid_mask = (depth_normalized > 0.01) & (depth_normalized < 0.99)
    
    # Extract point cloud data
    points = np.stack([X[valid_mask], Y[valid_mask], Z[valid_mask]], axis=1)
    colors = rgb_array[valid_mask] / 255.0  # Normalize colors to 0-1
    
    return points, colors

def load_image_and_depth(image_path, depth_path):
    """
    Load RGB image and depth image
    
    Args:
        image_path: Path to RGB image
        depth_path: Path to depth image
    
    Returns:
        rgb_image: RGB image (PIL Image)
        depth_image: Depth image (PIL Image)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth file not found: {depth_path}")
    
    rgb_image = Image.open(image_path).convert('RGB')
    depth_image = Image.open(depth_path).convert('L')  # Load as grayscale
    
    return rgb_image, depth_image

def downsample_pointcloud(points, colors, target_points=10000):
    """
    Downsample point cloud
    
    Args:
        points: Point cloud coordinates (N, 3)
        colors: Point cloud colors (N, 3)
        target_points: Target number of points
    
    Returns:
        downsampled_points: Downsampled point cloud coordinates
        downsampled_colors: Downsampled point cloud colors
    """
    if len(points) <= target_points:
        return points, colors
    
    # Random sampling
    indices = np.random.choice(len(points), target_points, replace=False)
    return points[indices], colors[indices] 