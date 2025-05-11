# ベースは CUDA 12.1 環境
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
ARG DEBIAN_FRONTEND=noninteractive

# 必要なAPTパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends \
    build-essential cmake git libeigen3-dev libopencv-dev libceres-dev \
    python3.10 python3.10-dev python3-pip python3.10-distutils \
    python-is-python3 curl ninja-build libglm-dev \
    libboost-all-dev libflann-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev \
    libglew-dev qtbase5-dev libqt5opengl5-dev wget libcgal-dev \
    graphviz mesa-utils libgraphviz-dev libgl1 libglib2.0-0 \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Python設定
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --config python3 --skip-auto && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    pip install --no-cache-dir --upgrade pip setuptools "setuptools<68.0.0"

# PyTorchとその他依存関係
RUN pip install --no-cache-dir torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir \
    numpy pandas matplotlib scipy seaborn \
    opencv-python-headless Pillow tqdm \
    flask loguru h5py scikit-learn jupyter jupyterlab \
    pyqtgraph pyyaml packaging pyparsing==3.0.9 networkx==2.5 \
    kornia==0.7.3 torchmetrics[image] imageio[ffmpeg] \
    black coverage mypy pylint pytest flake8 isort tyro>=0.8.8 \
    open3d colour tabulate simplejson parameterized pydegensac \
    tensorboard tensorly xmltodict cloudpickle==0.4.0 \
    fpdf2==2.4.6 python-dateutil Sphinx==4.2.0 \
    wheel viser nerfview \
    jsonschema==4.17.3 jupyter-events==0.6.3 \
    "PyOpenGL==3.1.1a1" "PyQt5" bokeh==2.4.3 \
    git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e \
    git+https://github.com/rahul-goel/fused-ssim

# splat_one 関連クローン＆ビルド
RUN git clone https://github.com/inuex35/splat_one.git /source/splat_one && \
    cd /source/splat_one && \
    git submodule update --init --recursive

RUN cd /source/splat_one/submodules/opensfm && \
    python3 setup.py build && \
    python3 setup.py install

RUN cd /source/splat_one/submodules/gsplat && \
    git checkout b0e978da67fb4364611c6683c5f4e6e6c1d8d8cb && \
    MAX_JOBS=4 pip install -e .

RUN sed -i 's/setuptools>=61.0/setuptools>=62.3.8,<75.9/' /source/splat_one/submodules/sam2/pyproject.toml && \
    cd /source/splat_one/submodules/sam2 && \
    pip install -e ".[notebooks]" && \
    cd checkpoints && ./download_ckpts.sh

# depth_any_camera のセットアップ
RUN git clone https://github.com/yuliangguo/depth_any_camera /depth_any_camera && \
    cd /depth_any_camera && \
    pip install -r requirements.txt

# PyTorchモデルの事前ダウンロード
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    wget https://download.pytorch.org/models/alexnet-owt-7be5be79.pth -O /root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth

# その他
RUN pip install mapillary_tools

WORKDIR /source/splat_one
