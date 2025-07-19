# splat_one

splat_one is an integrated application that combines Gaussian Splatting and Structure from Motion (SfM) into a single workflow. It allows users to visualize and process image data through multiple interactive tabs, making it easier to generate 3D point clouds and analyze camera parameters.

**Note:** This project is currently under development. The Masks tab is not functioning properly at this time.

## Tab Descriptions

- **Images Tab**  
  Displays information about the imported images, such as file names, resolutions, and metadata.

- **Masks Tab**  
  Provides functionality to generate masks for images.  
  *Currently under development; may not work as expected.*

- **Features Tab**  
  Shows the extracted image features, including keypoints and descriptors, for visualization and analysis.

- **Matching Tab**  
  Displays the results of feature point matching, allowing you to inspect the quality and accuracy of the correspondences.

- **Reconstruct Tab**  
  Visualizes the 3D point cloud reconstructed via SfM, along with camera positions and overall scene structure.

- **Gsplat Tab**  
  Presents the output of the Gaussian Splatting process. You can adjust parameters to explore different renderings of the 3D point cloud.

<p align="center">
  <a href="https://www.youtube.com/watch?v=m7eIe_ZGAqQ" target="_blank">
    <img src="https://github.com/user-attachments/assets/d6fe24f9-77b5-4a42-b879-c7c79144957d" alt="splat_one_resized" width="600"/>
  </a>
</p>

## Data Preparation

Before running the application, organize your images in the following directory structure inside the `dataset` folder:

```bash
splat_one
 └─ dataset
     └─ your_data
         └─ images
```

Place all the images you want to process under the `images` directory.

## Docker Deployment

This application is designed to run via Docker. The Docker image is available as `inuex35/splat_one`. Please refer to the `Dockerfile` for more details on the installation.

To launch the Docker container with GPU support and proper X11 forwarding (for GUI display), run the following command:

```bash
docker run --gpus all -e DISPLAY=host.docker.internal:0.0 -v /tmp/.X11-unix:/tmp/.X11-unix -v ${PWD}/dataset:/source/splat_one/dataset -v C:\Users\$env:USERNAME\.cache:/home/user/.cache/ -p 7007:7007 --rm -it --shm-size=12gb inuex35/splat_one
```

Once inside the container, start the application by executing:

```bash
python main.py
```

### Running with Depth Any Camera (DAC) Mode

To enable the Depth Any Camera feature for advanced depth estimation, you can run the application with the `--dac` flag:

```bash
docker run --gpus all -e DISPLAY=host.docker.internal:0.0 -v /tmp/.X11-unix:/tmp/.X11-unix -v ${PWD}/dataset:/source/splat_one/dataset -v C:\Users\$env:USERNAME\.cache:/home/user/.cache/ -p 7007:7007 --rm -it --shm-size=12gb inuex35/splat_one --dac
```

The `--dac` option enables depth estimation capabilities using the Depth Any Camera model, which provides robust depth prediction for various camera types including perspective, fisheye, and 360-degree cameras.

## Dependencies
- [OpenSfM](https://github.com/inuex35/ind-bermuda-opensfm/)
- [Gsplat](https://github.com/inuex35/gsplat/)

We use code from these repositories. Please note that the license for these components follows their respective original repositories.


# Development Status
The project is under active development.
Some features, such as the mask generation in the Masks tab, are not yet fully functional.
Contributions via Issues and Pull Requests are welcome.
License
This project is licensed under the MIT License. See the LICENSE file for details.
