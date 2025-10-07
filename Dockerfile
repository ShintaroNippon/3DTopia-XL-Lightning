FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu22.04

# Set environment variables for L4 GPU optimization
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX"
ENV FORCE_CUDA="1"
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libfontconfig1 \
    pkg-config \
    libglfw3-dev \
    libgles2-mesa-dev \
    libegl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements-fixed.txt .

# Upgrade pip and install core packages
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# CRITICAL: Install NumPy 1.x FIRST to avoid compatibility issues
RUN pip install --no-cache-dir "numpy<2.0.0" packaging

# Install PyTorch with CUDA support (L4 optimized)
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch3D (after PyTorch)
RUN pip install --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt212/download.html

# Install xformers for efficient attention (L4 compatible)
RUN pip install --no-cache-dir xformers==0.0.23

# Install other ML dependencies
RUN pip install --no-cache-dir \
    onnxruntime-gpu \
    triton==2.1.0 \
    transformers==4.40.1 \
    diffusers==0.19.3

# Install requirements
RUN pip install --no-cache-dir -r requirements-fixed.txt

# Copy the repository
COPY . .

# Create necessary directories
RUN mkdir -p pretrained inputs outputs logs cache

# Make scripts executable
RUN chmod +x *.sh scripts/*.sh install.sh

# Install 3DTopia dependencies with proper build order
RUN bash install.sh

# Create model download script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Downloading 3DTopia-XL models..."\n\
cd /workspace/pretrained\n\
\n\
# Download single-view model\n\
if [ ! -f "model_sview_dit_fp16.pt" ]; then\n\
    echo "Downloading single-view DiT model..."\n\
    wget -q --show-progress https://huggingface.co/FrozenBurning/3DTopia-XL/resolve/main/model_sview_dit_fp16.pt\n\
fi\n\
\n\
# Download VAE model\n\
if [ ! -f "model_vae_fp16.pt" ]; then\n\
    echo "Downloading VAE model..."\n\
    wget -q --show-progress https://huggingface.co/FrozenBurning/3DTopia-XL/resolve/main/model_vae_fp16.pt\n\
fi\n\
\n\
echo "Models downloaded successfully!"\n\
ls -la /workspace/pretrained/' > /workspace/download_models.sh && chmod +x /workspace/download_models.sh

# Create enhanced robot generation script
RUN echo '#!/usr/bin/env python3\n\
import os\n\
import sys\n\
import argparse\n\
import yaml\n\
from pathlib import Path\n\
\n\
def generate_robot(prompt="a futuristic robot", output_dir="/workspace/outputs"):\n\
    """Generate 3D robot with GLB export"""\n\
    \n\
    # Ensure output directory exists\n\
    os.makedirs(output_dir, exist_ok=True)\n\
    \n\
    # Create safe filename from prompt\n\
    safe_name = "".join(c for c in prompt if c.isalnum() or c in " -_").strip()\n\
    safe_name = safe_name.replace(" ", "_")[:50]\n\
    \n\
    output_path = os.path.join(output_dir, safe_name)\n\
    os.makedirs(output_path, exist_ok=True)\n\
    \n\
    # Create inference config\n\
    config = {\n\
        "model": {\n\
            "pretrained_model_path": "./pretrained/model_sview_dit_fp16.pt",\n\
            "vae_model_path": "./pretrained/model_vae_fp16.pt"\n\
        },\n\
        "inference": {\n\
            "prompt": prompt,\n\
            "output_dir": output_path,\n\
            "ddim_steps": 50,\n\
            "cfg_scale": 5.0,\n\
            "seed": 42,\n\
            "export_glb": True,\n\
            "mesh_resolution": 256\n\
        }\n\
    }\n\
    \n\
    config_path = os.path.join(output_path, "config.yml")\n\
    with open(config_path, "w") as f:\n\
        yaml.dump(config, f, default_flow_style=False)\n\
    \n\
    print(f"🚀 Generating: {prompt}")\n\
    print(f"📁 Output: {output_path}")\n\
    print(f"🔧 Config: {config_path}")\n\
    \n\
    # Run inference\n\
    cmd = f"cd /workspace && python inference.py {config_path}"\n\
    exit_code = os.system(cmd)\n\
    \n\
    if exit_code == 0:\n\
        print(f"✅ Success! Check {output_path} for GLB files")\n\
        # List generated files\n\
        for file in Path(output_path).glob("*.glb"):\n\
            print(f"📦 Generated: {file}")\n\
    else:\n\
        print(f"❌ Failed with exit code {exit_code}")\n\
    \n\
    return exit_code == 0\n\
\n\
if __name__ == "__main__":\n\
    parser = argparse.ArgumentParser(description="Generate 3D models with 3DTopia-XL")\n\
    parser.add_argument("--prompt", default="a futuristic robot", help="Text prompt")\n\
    parser.add_argument("--output", default="/workspace/outputs", help="Output directory")\n\
    args = parser.parse_args()\n\
    \n\
    generate_robot(args.prompt, args.output)' > /workspace/generate_3d.py && chmod +x /workspace/generate_3d.py

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import torch; print('GPU:', torch.cuda.is_available())" || exit 1

# Default startup
CMD ["/workspace/start_app.sh"]