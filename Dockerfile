FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX"
ENV FORCE_CUDA="1"
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
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
    libxrender1 \
    libxtst6 \
    libxi6 \
    libxrandr2 \
    libasound2 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Clone the original 3DTopia-XL repository
RUN git clone https://github.com/3DTopia/3DTopia-XL.git .

# CRITICAL: Fix NumPy compatibility issue
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install NumPy 1.x to avoid compatibility issues
RUN pip install "numpy<2.0.0"

# Install PyTorch with CUDA support (L4 optimized)
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install xformers for efficient attention
RUN pip install xformers==0.0.23

# Install onnxruntime for rembg (missing dependency)
RUN pip install onnxruntime-gpu

# Install requirements with NumPy compatibility
RUN pip install --no-cache-dir \
    "einops" \
    "omegaconf" \
    "opencv-python" \
    "libigl" \
    "trimesh==4.2.0" \
    "pygltflib" \
    "PyMCubes" \
    "xatlas" \
    "scikit-learn" \
    "open_clip_torch" \
    "triton==2.1.0" \
    "rembg[gpu]" \
    "gradio>=4.0.0" \
    "tqdm" \
    "transformers==4.40.1" \
    "diffusers==0.19.3" \
    "ninja" \
    "imageio" \
    "imageio-ffmpeg" \
    "gradio-litmodel3d==0.0.1" \
    "jaxtyping==0.2.31"

# Try to install pymeshlab (if it fails, we'll skip it)
RUN pip install pymeshlab==0.2 || echo "pymeshlab installation failed, continuing..."

# Install nvdiffrast from source
RUN pip install git+https://github.com/NVlabs/nvdiffrast/

# Additional dependencies for GLB export
RUN pip install --no-cache-dir \
    huggingface_hub \
    accelerate \
    safetensors \
    Pillow \
    scipy \
    matplotlib

# Compile third party libraries
RUN chmod +x install.sh && bash install.sh

# Create directories
RUN mkdir -p pretrained inputs outputs

# Create model download script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Downloading models for 3DTopia-XL..."\n\
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
# Download CLIP model\n\
if [ ! -f "open_clip_pytorch_model.bin" ]; then\n\
    echo "Downloading CLIP model..."\n\
    wget -q --show-progress "https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/resolve/main/open_clip_pytorch_model.bin?download=true" -O open_clip_pytorch_model.bin\n\
fi\n\
\n\
echo "All models downloaded successfully!"\n\
ls -la /workspace/pretrained/' > /workspace/download_models.sh && \
    chmod +x /workspace/download_models.sh

# Create enhanced robot generation script
RUN echo '#!/usr/bin/env python3\n\
import os\n\
import sys\n\
import argparse\n\
import subprocess\n\
import yaml\n\
from pathlib import Path\n\
\n\
def create_config(prompt, output_dir, ddim_steps=50, cfg_scale=5.0, seed=42):\n\
    """Create configuration for inference"""\n\
    config = {\n\
        "model": {\n\
            "checkpoint_path": "./pretrained/model_sview_dit_fp16.pt",\n\
            "vae_checkpoint_path": "./pretrained/model_vae_fp16.pt"\n\
        },\n\
        "inference": {\n\
            "input_dir": "./inputs",\n\
            "output_dir": output_dir,\n\
            "ddim": ddim_steps,\n\
            "cfg": cfg_scale,\n\
            "seed": seed,\n\
            "export_glb": True,\n\
            "fast_unwrap": False,\n\
            "decimate": 100000,\n\
            "mc_resolution": 256,\n\
            "remesh": False\n\
        },\n\
        "text_prompt": prompt\n\
    }\n\
    return config\n\
\n\
def main():\n\
    parser = argparse.ArgumentParser()\n\
    parser.add_argument("--prompt", default="a futuristic robot", help="Text prompt")\n\
    parser.add_argument("--output", default="outputs/futuristic_robot", help="Output directory")\n\
    parser.add_argument("--steps", type=int, default=50, help="DDIM steps")\n\
    parser.add_argument("--cfg", type=float, default=5.0, help="CFG scale")\n\
    parser.add_argument("--seed", type=int, default=42, help="Random seed")\n\
    \n\
    args = parser.parse_args()\n\
    \n\
    # Ensure models are downloaded\n\
    print("🔄 Ensuring models are downloaded...")\n\
    subprocess.run(["/workspace/download_models.sh"], check=True)\n\
    \n\
    # Create output directory\n\
    os.makedirs(args.output, exist_ok=True)\n\
    \n\
    # Create config\n\
    config = create_config(args.prompt, args.output, args.steps, args.cfg, args.seed)\n\
    config_path = os.path.join(args.output, "robot_config.yml")\n\
    \n\
    with open(config_path, "w") as f:\n\
        yaml.dump(config, f)\n\
    \n\
    print(f"🔧 Configuration saved to: {config_path}")\n\
    print(f"📁 Output directory: {args.output}")\n\
    print(f"🚀 Starting inference...")\n\
    \n\
    # Run inference\n\
    try:\n\
        result = subprocess.run(\n\
            ["python", "/workspace/inference.py", config_path],\n\
            check=True,\n\
            capture_output=True,\n\
            text=True\n\
        )\n\
        print("✅ Generation completed successfully!")\n\
        print(f"📂 Check {args.output} for GLB files")\n\
        \n\
        # List generated files\n\
        if os.path.exists(args.output):\n\
            files = list(Path(args.output).glob("*.glb"))\n\
            if files:\n\
                print("🎉 Generated GLB files:")\n\
                for f in files:\n\
                    print(f"   📦 {f}")\n\
    except subprocess.CalledProcessError as e:\n\
        print(f"❌ Generation failed with error code: {e.returncode}")\n\
        if e.stdout:\n\
            print("STDOUT:", e.stdout)\n\
        if e.stderr:\n\
            print("STDERR:", e.stderr)\n\
        print("💔 Generation failed. Please check the logs above.")\n\
        return 1\n\
    \n\
    return 0\n\
\n\
if __name__ == "__main__":\n\
    sys.exit(main())\n\
' > /workspace/enhanced_robot.py && chmod +x /workspace/enhanced_robot.py

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting 3DTopia-XL with NumPy compatibility fix..."\n\
\n\
# Download models\n\
/workspace/download_models.sh\n\
\n\
# Start Gradio interface\n\
python app.py --server_name 0.0.0.0 --server_port 7860 --share' > /workspace/start.sh && \
    chmod +x /workspace/start.sh

# Expose port
EXPOSE 7860

# Default command
CMD ["/workspace/start.sh"]