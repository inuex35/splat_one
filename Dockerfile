# Base image with CUDA support and Ubuntu 22.04
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

# Ensure UTF-8 locale settings
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
# Set non-interactive mode for apt-get
ARG DEBIAN_FRONTEND=noninteractive

# Install required apt packages
RUN apt-get update && apt install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa && apt list python3.* && apt-get install -y \
    build-essential \
    cmake \
    git \
    libeigen3-dev \
    libopencv-dev \
    libceres-dev \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-distutils \
    python-is-python3 \
    curl \
    ninja-build \
    libglm-dev \
    libboost-all-dev \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    wget \
    libcgal-dev \
    graphviz \
    mesa-utils \
    libgraphviz-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --config python3 --skip-auto

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

RUN pip install --upgrade pip setuptools

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Python dependencies directly
RUN pip install \
    black \
    coverage \
    mypy \
    pylint \
    pytest \
    flake8 \
    isort \
    dask[complete] \
    asyncssh \
    graphviz \
    matplotlib==3.4.2 \
    networkx \
    numpy \
    pandas \
    Pillow>=8.0.1 \
    scikit-learn \
    seaborn \
    scipy \
    hydra-core \
    pyparsing==3.0.9 \
    torch \
    torchvision>=0.13.0 \
    kornia==0.7.3 \
    pycolmap \
    gtsam==4.2 \
    h5py \
    plotly \
    tabulate \
    simplejson \
    parameterized \
    opencv-python>=4.5.4.58 \
    pydegensac \
    colour \
    "trimesh[easy]" \
    pydot \
    open3d \
    networkx==2.5 \
    packaging \
    tqdm \
    cloudpickle==0.4.0 \
    exifread==2.1.2 \
    flask \
    fpdf2==2.4.6 \
    joblib \
    "Pillow>=8.1.1" \
    pytest==3.0.7 \
    "python-dateutil>=2.7" \
    "scipy>=1.10.0" \
    Sphinx==4.2.0 \
    xmltodict==0.10.2 \
    wheel \
    viser \
    nerfview \
    "imageio[ffmpeg]" \
    "numpy<2.0.0" \
    tqdm \
    "torchmetrics[image]" \
    opencv-python \
    "tyro>=0.8.8" \
    Pillow \
    tensorboard \
    tensorly \
    pyyaml \
    matplotlib \
    pyqtgraph \
    loguru \
    PyOpenGL \
    PyOpenGL_accelerate \
    bokeh

ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# Install additional Python packages
RUN pip install git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e
RUN pip install git+https://github.com/rahul-goel/fused-ssim

# Clone and install FlashAttention
RUN git clone https://github.com/Dao-AILab/flash-attention.git /flash-attention && \
    cd /flash-attention && git checkout 6d48e14a6c2f551db96f0badc658a6279a929df3 && \
    pip install .


# Clone and install LightGlue
RUN git clone https://github.com/cvg/LightGlue.git /LightGlue && \
    cd /LightGlue && pip install .

RUN git clone https://github.com/inuex35/splat_one.git /source/splat_one && \
    cd /source/splat_one && \
    git submodule init && git submodule update --recursive

# Clone and build OpenSfM
RUN cd /source/splat_one/submodules/opensfm && \
    git submodule init && git submodule update --recursive && \
    python3 setup.py build && \
    python3 setup.py install && \
    pip install jupyter jupyterlab pyproj

# Add and build the gtsfm submodule
RUN cd /source/splat_one/submodules/gsplat && git checkout b0e978da67fb4364611c6683c5f4e6e6c1d8d8cb && MAX_JOBS=4 pip install -e .

RUN mkdir /source/splat_one/dataset
# Add and build the sam2 submodule
RUN cd /source/splat_one/submodules/sam2 && pip install -e ".[notebooks]" && cd checkpoints && ./download_ckpts.sh

RUN pip install --upgrade cloudpickle && pip install bokeh==2.4.3 && pip3 uninstall opencv-python -y && pip3 install opencv-python-headless && pip install PyOpenGL==3.1.1a1 PyQt5

# Set the working directory
WORKDIR /source/splat_one
