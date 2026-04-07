# CUDA 12.8 + cuDNN + 开发工具（适合编译 3D 检测 CUDA 扩展）
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC

WORKDIR /workspace

# ---- 系统依赖：Python、编译、常见运行库（OpenCV/可视化会用到）----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git curl wget ca-certificates \
    build-essential cmake ninja-build pkg-config \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# python -> python3
RUN ln -sf /usr/bin/python3 /usr/bin/python && python -m pip install -U pip setuptools wheel

# ---- 固定 PyTorch 版本（你指定的）----
RUN pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# ---- 额外构建依赖（spconv/cumm 编译用）----
    RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- 编译相关环境变量：锁定 5090 = sm_120，避免编译全家桶 ----
ENV CUDA_HOME=/usr/local/cuda \
    CUMM_CUDA_ARCH_LIST="12.0" \
    TORCH_CUDA_ARCH_LIST="12.0" \
    MAX_JOBS=8 \
    CMAKE_BUILD_PARALLEL_LEVEL=8

# ---- Python 侧编译工具链（pccm 是 cumm/spconv 体系的基础）----
RUN pip install -U \
    ninja \
    pccm \
    pybind11 \
    packaging \
    cmake

# ---- 拉源码并锁定到 tag（不要用 main；官方明确建议用 tags）----
# spconv 最新版是 2.3.8（PyPI 上标的是 2024-12-15）:contentReference[oaicite:4]{index=4}
# cumm 近期常用是 0.8.2（示例），并且官方提示开发/构建尽量用 tags :contentReference[oaicite:5]{index=5}
ARG SPCONV_TAG=v2.3.8
ARG CUMM_TAG=v0.8.2

RUN git clone --depth=1 --branch ${CUMM_TAG} https://github.com/FindDefinition/cumm.git /opt/cumm \
 && pip install -e /opt/cumm --no-build-isolation

 RUN git clone --depth=1 --branch ${SPCONV_TAG} https://github.com/traveller59/spconv.git /opt/spconv \
 && sed -i 's/<0.8.0/<0.9.0/g' /opt/spconv/setup.py \
 && pip install -e /opt/spconv --no-build-isolation --no-deps

# ---- 触发一次 import，让 JIT/编译在镜像构建期完成（避免运行时第一次 import 才编译）----
RUN python -c "import os; print('CUMM_CUDA_ARCH_LIST =', os.environ.get('CUMM_CUDA_ARCH_LIST')); print('TORCH_CUDA_ARCH_LIST =', os.environ.get('TORCH_CUDA_ARCH_LIST')); import cumm, spconv; print('cumm imported, spconv imported')"

# ---- 数值/训练/可视化常用依赖（建议适度锁版本，保证可复现）----
# 注：numpy/scipy 在 ODE/数值稳定性上很关键，建议锁住
RUN pip install \
    numpy==1.26.4 \
    scipy==1.11.4 \
    pandas==2.3.3 \
    tqdm==4.67.1 \
    pyyaml==6.0.3 \
    tensorboard==2.20.0 \
    matplotlib==3.10.8 \
    opencv-python~=4.11.0 \
    einops \
    omegaconf \
    hydra-core \
    h5py \
    pyquaternion \
    scikit-learn \
    scikit-image \
    rich \
    seaborn \
    plotly \
    torchdiffeq==0.2.5 \
    open3d==0.18.0 \
    torchcde

# 建议增大共享内存（运行时用 --shm-size=16g）
CMD ["/bin/bash"]

