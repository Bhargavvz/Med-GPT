# syntax=docker/dockerfile:1
# MedGPT — Full Stack Docker Image
# Stage 1: Build React frontend
# Stage 2: Run FastAPI + GPU inference

# ============================================
# Stage 1: Build Frontend
# ============================================
FROM node:20-slim AS frontend-build

WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ============================================
# Stage 2: Runtime with CUDA
# ============================================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

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

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY models/ ./models/
COPY training/ ./training/
COPY inference/ ./inference/
COPY data/dataset.py data/__init__.py ./data/
COPY data/processed/ ./data/processed/
COPY configs/ ./configs/
COPY checkpoints/ ./checkpoints/

# Copy frontend build from Stage 1
COPY --from=frontend-build /build/dist ./frontend/dist

# Create directories
RUN mkdir -p results data/images

# Expose port
EXPOSE 8000

# Environment
ENV MEDGPT_CONFIG=/app/configs/config.yaml
ENV HF_HOME=/app/.cache/huggingface

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Run server
CMD ["python", "backend/server.py"]
