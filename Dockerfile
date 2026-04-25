FROM tensorflow/tensorflow:2.14.0-gpu-jupyter

WORKDIR /workspace

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone repository
RUN git clone https://github.com/liang-wei-tan/face-mask-detection.git /workspace

# Install Python dependencies (tensorflow already installed in base image)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
