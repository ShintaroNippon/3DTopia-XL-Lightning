FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# Set environment variables optimized for L4 GPU
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX"
ENV FORCE_CUDA="1"
ENV CUDA_HOME=/usr/local/cuda
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

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
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Clone the original 3DTopia-XL repository
RUN git clone https://github.com/3DTopia/3DTopia-XL.git .

# Fix NumPy compatibility issue first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install "numpy<2.0.0" --force-reinstall

# Install PyTorch with CUDA support (L4 optimized)
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install xformers for efficient attention
RUN pip install xformers==0.0.23.post1

# Install missing dependencies first
RUN pip install --no-cache-dir \
    onnxruntime-gpu \
    onnx \
    protobuf \
    packaging

# Install requirements with NumPy constraint
RUN pip install --no-cache-dir \
    "numpy<2.0.0" \
    einops \
    omegaconf \
    opencv-python \
    libigl \
    "trimesh==4.2.0" \
    pygltflib \
    PyMCubes \
    xatlas \
    scikit-learn \
    open_clip_torch \
    rembg \
    gradio \
    tqdm \
    "transformers==4.40.1" \
    "diffusers==0.19.3" \
    ninja \
    imageio \
    imageio-ffmpeg \
    "gradio-litmodel3d==0.0.1" \
    "jaxtyping==0.2.31"

# Install pymeshlab separately (often causes issues)
RUN pip install pymeshlab==0.2 --no-deps || pip install pymeshlab

# Install triton with specific version for CUDA 11.8
RUN pip install triton==2.1.0 --no-deps

# Install nvdiffrast from source
RUN pip install git+https://github.com/NVlabs/nvdiffrast/

# Additional dependencies for GLB export
RUN pip install --no-cache-dir \
    huggingface_hub \
    accelerate \
    safetensors \
    Pillow \
    scipy \
    matplotlib \
    gltflib \
    meshio

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
ls -la /workspace/pretrained/\n\
echo "Ready for inference!"' > /workspace/download_models.sh && \
    chmod +x /workspace/download_models.sh

# Create enhanced robot generation script
RUN echo '#!/usr/bin/env python3\n\
import os\n\
import sys\n\
import subprocess\n\
import argparse\n\
from pathlib import Path\n\
import yaml\n\
\n\
def create_robot_config(prompt, output_dir, steps=50, cfg=5.0, seed=42):\n\
    """Create optimized config for robot generation with GLB export"""\n\
    config = {\n\
        "model": {\n\
            "target": "models.diffusion.DDPMSampler",\n\
            "params": {\n\
                "model_path": "./pretrained/model_sview_dit_fp16.pt",\n\
                "vae_path": "./pretrained/model_vae_fp16.pt",\n\
                "use_fp16": True\n\
            }\n\
        },\n\
        "inference": {\n\
            "prompt": prompt,\n\
            "output_dir": output_dir,\n\
            "ddim_steps": steps,\n\
            "cfg_scale": cfg,\n\
            "seed": seed,\n\
            "export_glb": True,\n\
            "fast_unwrap": False,\n\
            "decimate": 100000,\n\
            "mc_resolution": 256,\n\
            "remesh": False\n\
        }\n\
    }\n\
    return config\n\
\n\
def main():\n\
    parser = argparse.ArgumentParser(description="Generate futuristic robot 3D model")\n\
    parser.add_argument("--prompt", default="a futuristic robot", help="Text prompt")\n\
    parser.add_argument("--output", default="outputs/futuristic_robot", help="Output directory")\n\
    parser.add_argument("--steps", type=int, default=50, help="DDIM steps")\n\
    parser.add_argument("--cfg", type=float, default=5.0, help="CFG scale")\n\
    parser.add_argument("--seed", type=int, default=42, help="Random seed")\n\
    \n\
    args = parser.parse_args()\n\
    \n\
    print(f"🤖 Generating: {args.prompt}")\n\
    print(f"📁 Output: {args.output}")\n\
    \n\
    # Ensure models are downloaded\n\
    subprocess.run(["/workspace/download_models.sh"], check=True)\n\
    \n\
    # Create output directory\n\
    os.makedirs(args.output, exist_ok=True)\n\
    \n\
    # Create config\n\
    config = create_robot_config(args.prompt, args.output, args.steps, args.cfg, args.seed)\n\
    config_path = f"{args.output}/robot_config.yml"\n\
    \n\
    with open(config_path, "w") as f:\n\
        yaml.dump(config, f, default_flow_style=False)\n\
    \n\
    print(f"🔧 Configuration saved to: {config_path}")\n\
    print(f"📁 Output directory: {args.output}")\n\
    print(f"🚀 Starting inference...")\n\
    \n\
    try:\n\
        # Run inference\n\
        result = subprocess.run([\n\
            "python", "inference.py", "./configs/inference_dit.yml"\n\
        ], capture_output=True, text=True, cwd="/workspace")\n\
        \n\
        if result.returncode == 0:\n\
            print("✅ Generation completed successfully!")\n\
            print(f"📂 Check {args.output} for GLB files")\n\
            \n\
            # List generated files\n\
            if os.path.exists(args.output):\n\
                files = list(Path(args.output).glob("*.glb"))\n\
                if files:\n\
                    print("🎉 Generated GLB files:")\n\
                    for file in files:\n\
                        print(f"   📦 {file}")\n\
                else:\n\
                    print("⚠️  No GLB files found, but generation completed")\n\
        else:\n\
            print(f"❌ Generation failed with error code: {result.returncode}")\n\
            print(f"stderr: {result.stderr}")\n\
            return False\n\
            \n\
    except Exception as e:\n\
        print(f"❌ Generation failed with exception: {e}")\n\
        return False\n\
    \n\
    return True\n\
\n\
if __name__ == "__main__":\n\
    success = main()\n\
    if success:\n\
        print("\\n🎊 Robot generation completed successfully!")\n\
    else:\n\
        print("\\n💔 Generation failed. Please check the logs above.")\n\
        sys.exit(1)' > /workspace/enhanced_robot.py && \
    chmod +x /workspace/enhanced_robot.py

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "🚀 Starting 3DTopia-XL on Lightning.ai..."\n\
\n\
# Download models\n\
/workspace/download_models.sh\n\
\n\
# Check GPU\n\
echo "GPU Info:"\n\
nvidia-smi || echo "No GPU detected"\n\
\n\
# Start Gradio interface\n\
echo "Starting Gradio interface on port 7860..."\n\
python app.py --server_name 0.0.0.0 --server_port 7860 --share' > /workspace/start.sh && \
    chmod +x /workspace/start.sh

# Set default command
CMD ["/workspace/start.sh"]

# Expose port
EXPOSE 7860