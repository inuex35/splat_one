# Base image with CUDA support and Ubuntu 22.04
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

# Ensure UTF-8 locale settings
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Set non-interactive mode for apt-get
ARG DEBIAN_FRONTEND=noninteractive

# Install required apt packages (キャッシュを残さない)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends \
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
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --config python3 --skip-auto

# Install pip via get-pip.py and upgrade without caching
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    pip install --no-cache-dir --upgrade pip setuptools
RUN pip install --upgrade pip setuptools

# Install PyTorch packages with no cache
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install additional Python dependencies (--no-cache-dir を追加)
RUN pip install --no-cache-dir \
    black \
    coverage \
    mypy \
    pylint \
    pytest \
    flake8 \
    isort \
    "dask[complete]" \
    asyncssh \
    graphviz \
    "matplotlib==3.4.2" \
    networkx==2.5 \
    numpy \
    pandas \
    "Pillow>=8.0.1" \
    scikit-learn \
    seaborn \
    scipy \
    hydra-core \
    "pyparsing==3.0.9" \
    torch \
    "torchvision>=0.13.0" \
    "kornia==0.7.3" \
    pycolmap \
    "gtsam==4.2" \
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
    packaging \
    tqdm \
    "cloudpickle==0.4.0" \
    "exifread==2.1.2" \
    flask \
    "fpdf2==2.4.6" \
    joblib \
    "python-dateutil>=2.7" \
    "scipy>=1.10.0" \
    "Sphinx==4.2.0" \
    "xmltodict==0.10.2" \
    wheel \
    viser \
    nerfview \
    "imageio[ffmpeg]" \
    "numpy<2.0.0" \
    "torchmetrics[image]" \
    opencv-python \
    "tyro>=0.8.8" \
    tensorboard \
    tensorly \
    pyyaml \
    pyqtgraph \
    loguru \
    PyOpenGL \
    PyOpenGL_accelerate \
    bokeh \
    --no-cache-dir

# CUDA architecture
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# Install additional Python packages from Git repositories without cache
RUN pip install --no-cache-dir git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e && \
    pip install --no-cache-dir git+https://github.com/rahul-goel/fused-ssim

# Clone and install FlashAttention with shallow clone and remove .git afterwards
RUN git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /flash-attention && \
    cd /flash-attention && \
    pip install --no-cache-dir .
    
# Clone and install LightGlue with shallow clone
RUN git clone --depth 1 https://github.com/cvg/LightGlue.git /LightGlue && \
    cd /LightGlue && pip install --no-cache-dir .

# Clone splat_one リポジトリも浅いクローンに変更し、サブモジュールも更新後に不要な .git を削除
RUN git clone https://github.com/inuex35/splat_one.git /source/splat_one && \
    cd /source/splat_one && \
    git submodule update --init --recursive

# OpenSfM のビルドとインストール（ビルド後にキャッシュ削除）
RUN cd /source/splat_one/submodules/opensfm && \
    git submodule update --init --recursive && \
    python3 setup.py build && \
    python3 setup.py install && \
    pip install --no-cache-dir jupyter jupyterlab pyproj

# gtsfm サブモジュールのビルド
RUN cd /source/splat_one/submodules/gsplat && \
    git checkout b0e978da67fb4364611c6683c5f4e6e6c1d8d8cb && \
    MAX_JOBS=4 pip install -e .

RUN mkdir /source/splat_one/dataset

# sam2 サブモジュールのビルド
RUN cd /source/splat_one/submodules/sam2 && \
    pip install -e ".[notebooks]" && \
    cd checkpoints && ./download_ckpts.sh

# opencv-python の入れ替えと他パッケージのインストール
RUN pip install --no-cache-dir --upgrade cloudpickle && \
    pip install --no-cache-dir bokeh==2.4.3 && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir opencv-python-headless && \
    pip install --no-cache-dir "PyOpenGL==3.1.1a1" "PyQt5"

RUN mkdir -p /root/.cache/torch/hub/checkpoints && \ wget https://download.pytorch.org/models/alexnet-owt-7be5be79.pth -O /root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth

# Set the working directory
WORKDIR /source/splat_one
