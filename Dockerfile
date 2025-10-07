FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=all
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX"
ENV FORCE_CUDA="1"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libglfw3-dev \
    libgles2-mesa-dev \
    libegl1-mesa-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Clone 3DTopia-XL source code
RUN git clone https://github.com/3DTopia/3DTopia-XL.git . && \
    rm -rf .git

# Copy your custom files
COPY requirements-fixed.txt ./
COPY *.py ./
COPY *.sh ./

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# CRITICAL: Install NumPy 1.x FIRST to avoid compatibility issues
RUN pip3 install --no-cache-dir "numpy<2.0.0"

# Install PyTorch with CUDA 11.8 support
RUN pip3 install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch3D (after PyTorch is installed)
RUN pip3 install --no-cache-dir \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install xformers with CUDA 11.8 compatibility
RUN pip3 install --no-cache-dir xformers==0.0.23

# Install other ML dependencies
RUN pip3 install --no-cache-dir \
    onnxruntime-gpu \
    triton==2.1.0 \
    transformers==4.40.1 \
    diffusers==0.19.3

# Install requirements
RUN pip3 install --no-cache-dir -r requirements-fixed.txt

# Install nvdiffrast
RUN pip3 install git+https://github.com/NVlabs/nvdiffrast/

# Make scripts executable
RUN chmod +x *.sh

# Compile CUDA extensions
RUN bash install.sh || echo "Some extensions failed to compile, continuing..."

# Create necessary directories
RUN mkdir -p pretrained inputs outputs logs cache

# Create model download script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Downloading 3DTopia-XL models..."\n\
cd /workspace/pretrained\n\
\n\
if [ ! -f "model_sview_dit_fp16.pt" ]; then\n\
    echo "Downloading single-view DiT model..."\n\
    wget -q --show-progress https://huggingface.co/FrozenBurning/3DTopia-XL/resolve/main/model_sview_dit_fp16.pt\n\
fi\n\
\n\
if [ ! -f "model_vae_fp16.pt" ]; then\n\
    echo "Downloading VAE model..."\n\
    wget -q --show-progress https://huggingface.co/FrozenBurning/3DTopia-XL/resolve/main/model_vae_fp16.pt\n\
fi\n\
\n\
echo "Models downloaded successfully!"\n\
ls -la /workspace/pretrained/' > /workspace/download_models.sh && chmod +x /workspace/download_models.sh

# Create enhanced inference script for GLB generation
RUN echo '#!/usr/bin/env python3\n\
import os\n\
import sys\n\
import argparse\n\
import subprocess\n\
from pathlib import Path\n\
\n\
def generate_glb(prompt, output_dir="/workspace/outputs"):\n\
    """Generate 3D model from prompt with GLB export"""\n\
    \n\
    print(f"🚀 Generating GLB for: {prompt}")\n\
    \n\
    # Ensure models are downloaded\n\
    os.system("/workspace/download_models.sh")\n\
    \n\
    # Create safe output directory\n\
    safe_name = "".join(c for c in prompt if c.isalnum() or c in " -_").strip()\n\
    safe_name = safe_name.replace(" ", "_")[:50]\n\
    \n\
    full_output_dir = os.path.join(output_dir, safe_name)\n\
    os.makedirs(full_output_dir, exist_ok=True)\n\
    \n\
    print(f"📁 Output directory: {full_output_dir}")\n\
    \n\
    # Set environment variables for inference\n\
    env = os.environ.copy()\n\
    env["PROMPT"] = prompt\n\
    env["OUTPUT_DIR"] = full_output_dir\n\
    env["CUDA_VISIBLE_DEVICES"] = "0"\n\
    \n\
    try:\n\
        # Run inference with the original script\n\
        result = subprocess.run([\n\
            "python", "/workspace/inference.py",\n\
            "/workspace/configs/inference_dit.yml"\n\
        ], env=env, cwd="/workspace", capture_output=True, text=True)\n\
        \n\
        if result.returncode == 0:\n\
            print("✅ Generation completed successfully!")\n\
            \n\
            # List generated files\n\
            output_files = list(Path(full_output_dir).glob("*"))\n\
            glb_files = [f for f in output_files if f.suffix.lower() == ".glb"]\n\
            \n\
            if glb_files:\n\
                print(f"📦 Generated {len(glb_files)} GLB files:")\n\
                for glb_file in glb_files:\n\
                    size_mb = glb_file.stat().st_size / (1024 * 1024)\n\
                    print(f"   🔗 {glb_file.name} ({size_mb:.1f} MB)")\n\
            else:\n\
                print(f"📄 Generated {len(output_files)} files (no GLB found)")\n\
                for f in output_files:\n\
                    print(f"   📄 {f.name}")\n\
            \n\
            return True\n\
        else:\n\
            print(f"❌ Generation failed")\n\
            print(f"Error output: {result.stderr}")\n\
            return False\n\
            \n\
    except Exception as e:\n\
        print(f"💥 Error during generation: {e}")\n\
        return False\n\
\n\
if __name__ == "__main__":\n\
    parser = argparse.ArgumentParser(description="Generate GLB files from text prompts")\n\
    parser.add_argument("--prompt", required=True, help="Text prompt for 3D generation")\n\
    parser.add_argument("--output_dir", default="/workspace/outputs", help="Output directory")\n\
    args = parser.parse_args()\n\
    \n\
    success = generate_glb(args.prompt, args.output_dir)\n\
    sys.exit(0 if success else 1)' > /workspace/inference_glb.py && chmod +x /workspace/inference_glb.py

# Expose port
EXPOSE 7860

# Default command
CMD ["/workspace/download_models.sh"]