# syntax=docker/dockerfile:1
# MedGPT — Medical Visual Question Answering System
# Docker image with NVIDIA CUDA support for GPU inference

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create app directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data and checkpoint directories
RUN mkdir -p data/processed data/images checkpoints

# Expose web application port
EXPOSE 8000

# Environment variables
ENV MEDGPT_CONFIG=/app/configs/config.yaml
ENV HF_HOME=/app/.cache/huggingface

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default: run the web application
CMD ["python", "app/server.py"]
