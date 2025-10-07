#!/bin/bash
set -e

echo "🚀 Starting 3DTopia-XL on Lightning.ai..."

# Download models
/workspace/download_models.sh

# Set GPU optimizations for L4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_LAUNCH_BLOCKING=0

# Check GPU
python -c "import torch; print('🔥 CUDA available:', torch.cuda.is_available()); print('🎮 GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "✅ Starting Gradio interface on port 7860..."
cd /workspace
python app.py --server_name 0.0.0.0 --server_port 7860 --share