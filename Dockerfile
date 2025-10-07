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

# Clone the original 3DTopia-XL repository
RUN git clone https://github.com/3DTopia/3DTopia-XL.git temp && \
    cp -r temp/* . && \
    rm -rf temp

# Copy your custom files
COPY requirements-fixed.txt .
COPY start_app.sh .
COPY lightning.yaml .

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
RUN pip install --no-cache-dir pytorch3d

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

# Create necessary directories
RUN mkdir -p pretrained inputs outputs logs cache

# Make scripts executable
RUN chmod +x *.sh

# Install 3DTopia dependencies with proper build order
RUN bash install.sh || echo "Some dependencies failed, continuing..."

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

# Create prompt-based inference script
RUN echo '#!/usr/bin/env python3\n\
import os\n\
import sys\n\
import argparse\n\
import subprocess\n\
from pathlib import Path\n\
\n\
def generate_from_prompt(prompt, output_dir="outputs"):\n\
    """Generate 3D model from text prompt with GLB export"""\n\
    \n\
    # Create safe directory name from prompt\n\
    safe_name = "".join(c for c in prompt if c.isalnum() or c in " -_").strip()\n\
    safe_name = safe_name.replace(" ", "_")[:50]\n\
    \n\
    output_path = os.path.join(output_dir, safe_name)\n\
    os.makedirs(output_path, exist_ok=True)\n\
    \n\
    print(f"🚀 Generating: {prompt}")\n\
    print(f"📁 Output: {output_path}")\n\
    \n\
    # Set environment for inference\n\
    env = os.environ.copy()\n\
    env["PROMPT"] = prompt\n\
    env["OUTPUT_DIR"] = output_path\n\
    env["EXPORT_GLB"] = "true"\n\
    \n\
    # Run inference using the original script\n\
    try:\n\
        result = subprocess.run([\n\
            "python", "inference.py", \n\
            "./configs/inference_dit.yml"\n\
        ], env=env, cwd="/workspace", capture_output=True, text=True)\n\
        \n\
        if result.returncode == 0:\n\
            print("✅ Generation completed successfully!")\n\
            \n\
            # List generated files\n\
            output_files = list(Path(output_path).glob("*"))\n\
            if output_files:\n\
                print(f"📦 Generated {len(output_files)} files:")\n\
                for file in output_files:\n\
                    print(f"   - {file.name}")\n\
            \n\
            return True\n\
        else:\n\
            print(f"❌ Generation failed")\n\
            print(f"Error: {result.stderr}")\n\
            return False\n\
            \n\
    except Exception as e:\n\
        print(f"💥 Error: {e}")\n\
        return False\n\
\n\
if __name__ == "__main__":\n\
    parser = argparse.ArgumentParser(description="Generate 3D models from prompts")\n\
    parser.add_argument("--prompt", required=True, help="Text prompt for generation")\n\
    parser.add_argument("--output", default="outputs", help="Output directory")\n\
    args = parser.parse_args()\n\
    \n\
    # Download models first\n\
    os.system("/workspace/download_models.sh")\n\
    \n\
    success = generate_from_prompt(args.prompt, args.output)\n\
    sys.exit(0 if success else 1)' > /workspace/generate_prompt.py && chmod +x /workspace/generate_prompt.py

# Expose port
EXPOSE 7860

# Default startup
CMD ["/workspace/start_app.sh"]