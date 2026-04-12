# Use NVIDIA CUDA 12.8 with cuDNN runtime
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

WORKDIR /workspace

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Clone repository
RUN git clone https://github.com/liang-wei-tan/face-mask-detection.git /workspace

# Install Python dependencies
RUN pip install -r requirements.txt

# Install Jupyter for development
RUN pip install jupyterlab

# Expose Jupyter port
EXPOSE 8888

# Start JupyterLab on container startup
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
