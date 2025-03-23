FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    python3-opengl \
    python3-pyqt5 \
    python3-pyqt5.qtopengl \
    python3-numpy \
    python3-scipy \
    python3-tk \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    ffmpeg \
    wget \
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /source/splat_one

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install sphinx for OpenSfM documentation
RUN pip3 install sphinx sphinx-rtd-theme

# Install OpenSfM
RUN git clone https://github.com/mapillary/OpenSfM.git /source/OpenSfM \
    && cd /source/OpenSfM \
    && pip3 install -e .

# Install SAM2 and download the checkpoint
RUN git clone https://github.com/facebookresearch/segment-anything-2.git /source/sam2 \
    && cd /source/sam2 \
    && pip3 install -e . \
    && mkdir -p checkpoints \
    && wget -P checkpoints https://huggingface.co/lkeab/hiera-sam-vit/resolve/main/sam2.1_hiera_large.pt

# Install mapillary_tools for video processing
RUN pip3 install mapillary_tools

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p config
RUN mkdir -p configs/sam2.1
RUN mkdir -p /root/.config/mapillary

# Set Python path to include SAM2
ENV PYTHONPATH=/source/sam2:$PYTHONPATH

# Set display for GUI application
ENV DISPLAY=:0

# Command to run when the container starts
CMD ["python3", "app.py"]
