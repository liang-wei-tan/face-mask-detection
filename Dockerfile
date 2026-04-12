FROM tensorflow/tensorflow:2.14.0-gpu-jupyter

WORKDIR /workspace

# Install additional system dependencies including openssh-server for direct SSH
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Clone repository
RUN git clone https://github.com/liang-wei-tan/face-mask-detection.git /workspace

# Install Python dependencies (tensorflow already installed in base image)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose Jupyter port
EXPOSE 8888
