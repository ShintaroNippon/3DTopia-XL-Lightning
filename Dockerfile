FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# Set environment variables optimized for L4 GPU
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX"
ENV FORCE_CUDA="1"
ENV CUDA_HOME=/usr/local/cuda
ENV TCNN_CUDA_ARCHITECTURES="75;80;86"

# Install system dependencies
RUN apt-get update && apt-get install -y \
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
    libfontconfig1 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libxrandr2 \
    libasound2 \
    pkg-config \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy repository files
COPY . .

# Install Python dependencies in correct order to avoid NumPy conflicts
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Fix NumPy version FIRST (critical for compatibility)
RUN pip install "numpy<2.0.0,>=1.21.0"

# Install PyTorch with CUDA support (L4 optimized)
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install xformers for efficient attention
RUN pip install xformers==0.0.23.post1

# Install specific problematic dependencies first
RUN pip install --no-cache-dir \
    onnxruntime-gpu==1.16.3 \
    ninja \
    packaging \
    wheel

# Install core ML dependencies
RUN pip install --no-cache-dir \
    einops \
    omegaconf \
    opencv-python \
    scikit-learn \
    open_clip_torch \
    tqdm \
    transformers==4.40.1 \
    diffusers==0.19.3 \
    imageio \
    imageio-ffmpeg \
    jaxtyping==0.2.31 \
    rembg \
    gradio>=4.0.0

# Install 3D processing libraries (with specific versions for compatibility)
RUN pip install --no-cache-dir \
    "trimesh==4.2.0" \
    pygltflib \
    PyMCubes \
    xatlas \
    libigl

# Install pymeshlab separately (often causes issues)
RUN pip install --no-cache-dir pymeshlab==0.2.1a10 || \
    pip install --no-cache-dir pymeshlab==2022.2.post4 || \
    echo "Warning: pymeshlab installation failed, continuing..."

# Install triton for L4 GPU
RUN pip install triton==2.1.0 || \
    pip install triton==2.0.0 || \
    echo "Warning: triton installation failed, continuing..."

# Install gradio-litmodel3d
RUN pip install --no-cache-dir gradio-litmodel3d==0.0.1 || \
    echo "Warning: gradio-litmodel3d installation failed, continuing..."

# Install nvdiffrast from source
RUN pip install git+https://github.com/NVlabs/nvdiffrast/

# Additional dependencies for Lightning.ai and GLB export
RUN pip install --no-cache-dir \
    huggingface_hub \
    accelerate \
    safetensors \
    Pillow>=8.0.0 \
    scipy \
    matplotlib \
    requests \
    aiohttp

# Compile third party libraries
RUN chmod +x install.sh && bash install.sh

# Create directories
RUN mkdir -p pretrained inputs outputs logs cache

# Create model download script
RUN echo '#!/bin/bash\n\
echo "Downloading models for 3DTopia-XL (L4 optimized)..."\n\
cd /workspace/pretrained\n\
\n\
# Download single-view model (FP16 for L4 efficiency)\n\
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
# Download CLIP model for text conditioning\n\
if [ ! -f "open_clip_pytorch_model.bin" ]; then\n\
    echo "Downloading CLIP model..."\n\
    wget -q --show-progress "https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/resolve/main/open_clip_pytorch_model.bin?download=true" -O open_clip_pytorch_model.bin\n\
fi\n\
\n\
echo "All models downloaded successfully!"\n\
echo "Total model size:"\n\
du -sh /workspace/pretrained/\n\
ls -la /workspace/pretrained/' > /workspace/download_models.sh && \
    chmod +x /workspace/download_models.sh

# Create enhanced robot generation script
RUN echo '#!/usr/bin/env python3\n\
"""\n\
Enhanced robot generation script for Lightning.ai L4 GPU\n\
Fixes NumPy compatibility and includes GLB export\n\
"""\n\
\n\
import os\n\
import sys\n\
import yaml\n\
import argparse\n\
from pathlib import Path\n\
\n\
# Fix NumPy compatibility before importing other modules\n\
import numpy as np\n\
print(f"Using NumPy version: {np.__version__}")\n\
\n\
# Set environment variables for L4 optimization\n\
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"\n\
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"\n\
\n\
def create_robot_config(prompt, output_dir, steps=50, cfg=5.0, seed=42):\n\
    """Create configuration for robot generation with GLB export"""\n\
    \n\
    config = {\n\
        \"model\": {\n\
            \"target\": \"models.dit_models.DiT3D_Latte\",\n\
            \"params\": {\n\
                \"input_size\": 64,\n\
                \"patch_size\": 2,\n\
                \"in_channels\": 16,\n\
                \"hidden_size\": 1152,\n\
                \"depth\": 28,\n\
                \"num_heads\": 16,\n\
                \"mlp_ratio\": 4.0,\n\
                \"class_dropout_prob\": 0.1,\n\
                \"num_classes\": 1000,\n\
                \"learn_sigma\": True\n\
            }\n\
        },\n\
        \"vae\": {\n\
            \"target\": \"models.vae_models.AutoencoderKL\",\n\
            \"params\": {\n\
                \"embed_dim\": 16\n\
            }\n\
        },\n\
        \"inference\": {\n\
            \"prompt\": prompt,\n\
            \"ddim_steps\": steps,\n\
            \"cfg_scale\": cfg,\n\
            \"seed\": seed,\n\
            \"export_glb\": True,\n\
            \"fast_unwrap\": False,\n\
            \"decimate\": 100000,\n\
            \"mc_resolution\": 256,\n\
            \"remesh\": False,\n\
            \"output_dir\": output_dir\n\
        },\n\
        \"paths\": {\n\
            \"pretrained_model_path\": \"pretrained/model_sview_dit_fp16.pt\",\n\
            \"vae_model_path\": \"pretrained/model_vae_fp16.pt\",\n\
            \"clip_model_path\": \"pretrained/open_clip_pytorch_model.bin\"\n\
        }\n\
    }\n\
    \n\
    return config\n\
\n\
def main():\n\
    parser = argparse.ArgumentParser(description=\"Generate futuristic robot with GLB export\")\n\
    parser.add_argument(\"--prompt\", default=\"a futuristic robot\", help=\"Text prompt\")\n\
    parser.add_argument(\"--output_dir\", default=\"/workspace/outputs/futuristic_robot\", help=\"Output directory\")\n\
    parser.add_argument(\"--steps\", type=int, default=50, help=\"DDIM steps\")\n\
    parser.add_argument(\"--cfg\", type=float, default=5.0, help=\"CFG scale\")\n\
    parser.add_argument(\"--seed\", type=int, default=42, help=\"Random seed\")\n\
    \n\
    args = parser.parse_args()\n\
    \n\
    print(f\"🤖 Generating: {args.prompt}\")\n\
    print(f\"📁 Output: {args.output_dir}\")\n\
    print(f\"⚙️  Settings: steps={args.steps}, cfg={args.cfg}, seed={args.seed}\")\n\
    \n\
    # Ensure models are downloaded\n\
    print(\"📥 Ensuring models are available...\")\n\
    os.system(\"/workspace/download_models.sh\")\n\
    \n\
    # Create output directory\n\
    output_path = Path(args.output_dir)\n\
    output_path.mkdir(parents=True, exist_ok=True)\n\
    \n\
    # Create config\n\
    config = create_robot_config(args.prompt, args.output_dir, args.steps, args.cfg, args.seed)\n\
    config_path = output_path / \"robot_config.yml\"\n\
    \n\
    with open(config_path, \"w\") as f:\n\
        yaml.dump(config, f, default_flow_style=False)\n\
    \n\
    print(f\"🔧 Configuration saved to: {config_path}\")\n\
    print(f\"📁 Output directory: {args.output_dir}\")\n\
    print(f\"🚀 Starting inference...\")\n\
    \n\
    # Run inference\n\
    try:\n\
        import subprocess\n\
        result = subprocess.run([\n\
            \"python\", \"inference.py\", str(config_path)\n\
        ], check=True, capture_output=True, text=True, cwd=\"/workspace\")\n\
        \n\
        print(\"✅ Generation completed successfully!\")\n\
        \n\
        # List generated files\n\
        output_files = list(output_path.glob(\"*\"))\n\
        if output_files:\n\
            print(f\"🎉 Generated files:\")\n\
            for file_path in output_files:\n\
                print(f\"   📦 {file_path.name} ({file_path.stat().st_size // 1024} KB)\")\n\
        \n\
        # Look for GLB files specifically\n\
        glb_files = list(output_path.glob(\"*.glb\"))\n\
        if glb_files:\n\
            print(f\"\\n🎊 GLB files ready for download:\")\n\
            for glb_file in glb_files:\n\
                print(f\"   🔗 {glb_file}\")\n\
        \n\
        return True\n\
        \n\
    except subprocess.CalledProcessError as e:\n\
        print(f\"❌ Generation failed with error code: {e.returncode}\")\n\
        if e.stdout:\n\
            print(f\"stdout: {e.stdout}\")\n\
        if e.stderr:\n\
            print(f\"stderr: {e.stderr}\")\n\
        return False\n\
    except Exception as e:\n\
        print(f\"💥 Unexpected error: {e}\")\n\
        return False\n\
\n\
if __name__ == \"__main__\":\n\
    success = main()\n\
    if success:\n\
        print(f\"\\n🎊 Successfully generated your futuristic robot!\")\n\
        print(f\"📂 Check the output directory for GLB files!\")\n\
    else:\n\
        print(f\"\\n💔 Generation failed. Please check the logs above.\")\n\
        sys.exit(1)\n\
' > /workspace/enhanced_robot.py && chmod +x /workspace/enhanced_robot.py

# Create startup script for Lightning.ai
RUN echo '#!/bin/bash\n\
echo \"🚀 Starting 3DTopia-XL on Lightning.ai L4 GPU...\"\n\
\n\
# Set GPU optimizations\n\
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256\n\
export CUDA_LAUNCH_BLOCKING=0\n\
\n\
# Download models\n\
echo \"📥 Downloading models...\"\n\
/workspace/download_models.sh\n\
\n\
# Show GPU info\n\
echo \"🖥️  GPU Information:\"\n\
nvidia-smi\n\
\n\
echo \"🌐 Starting Gradio interface...\"\n\
echo \"Access through Lightning.ai port forwarding on port 7860\"\n\
echo \"\"\n\
echo \"💡 For CLI robot generation:\"\n\
echo \"python enhanced_robot.py --prompt \\\"a futuristic robot\\\"\"\n\
echo \"\"\n\
\n\
# Start Gradio app\n\
cd /workspace\n\
python app.py --server_name 0.0.0.0 --server_port 7860 --share' > /workspace/start_lightning.sh && \
    chmod +x /workspace/start_lightning.sh

# Set default command
CMD ["/workspace/start_lightning.sh"]

# Expose port
EXPOSE 7860