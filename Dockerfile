FROM tensorflow/tensorflow:2.14.0-gpu-jupyter

WORKDIR /workspace

# Install additional system dependencies including SSH and nginx
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    openssh-server \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Remove SSH host keys so RunPod can generate them at startup
RUN rm -f /etc/ssh/ssh_host_*

# Install Python dependencies (tensorflow already installed in base image)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy startup script
COPY start.sh /
RUN chmod +x /start.sh

# Start SSH and JupyterLab
CMD ["/start.sh"]
