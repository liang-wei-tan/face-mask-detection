#!/bin/bash

# Generate SSH host keys if they don't exist
if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
    ssh-keygen -A
fi

# Start SSH daemon
service ssh start

# Start JupyterLab
jupyter lab --ip=0.0.0.0 --allow-root --no-browser
