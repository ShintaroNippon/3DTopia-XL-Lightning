FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# Set environment variables for L4 GPU optimization
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=all
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies including those needed for pymeshlab
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    ffmpeg \
    libglfw3-dev \
    libgles2-mesa-dev \
    libegl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt requirements-fixed.txt ./

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir -r requirements-fixed.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p models datasets outputs logs

# Set environment for the app
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose port for the app
EXPOSE 7860

# Set the default command
CMD ["python3", "app.py"]