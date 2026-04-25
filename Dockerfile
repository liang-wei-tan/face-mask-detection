FROM tensorflow/tensorflow:2.14.0-gpu-jupyter

WORKDIR /workspace

# Install additional system dependencies including openssh-server for direct SSH
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH directory and permissions (required for RunPod SSH key injection)
RUN mkdir -p /root/.ssh /run/sshd && \
    chmod 700 /root/.ssh && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/PermitEmptyPasswords no/PermitEmptyPasswords no/' /etc/ssh/sshd_config

# Clone repository
RUN git clone https://github.com/liang-wei-tan/face-mask-detection.git /workspace

# Install Python dependencies (tensorflow already installed in base image)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Start SSH and JupyterLab
CMD service ssh start && jupyter lab --ip=0.0.0.0 --allow-root --no-browser
