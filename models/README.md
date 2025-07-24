# Depth Estimation Models

This directory should contain the pre-trained models for depth estimation.

## Depth Anything V2 Models

Download the models from the official repository:
- https://github.com/DepthAnything/Depth-Anything-V2

### Recommended models:
- **depth_anything_v2_vitl.pth** - Large model (best quality)
- **depth_anything_v2_vits.pth** - Small model (faster inference)

Download links:
- ViT-L: https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
- ViT-S: https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth

## DAC (Depth Anything Camera) Models

Download the models from:
- https://github.com/xanderchf/depth_any_camera

### Available models:
- **dac_vitl_hypersim.pth** - Trained on HyperSim dataset
- **dac_vitl_vkitti.pth** - Trained on Virtual KITTI dataset

Choose based on your use case:
- HyperSim: Better for indoor scenes
- VKITTI: Better for outdoor/driving scenes

## Installation

1. Create this models directory if it doesn't exist
2. Download the desired model files
3. Place them in this directory
4. The depth tab will automatically detect and use them

## Note

The models are large files (several hundred MB each). Make sure you have enough disk space.