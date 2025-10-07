#!/bin/bash
set -e

echo "=== Building 3DTopia-XL with NumPy Fix ==="

# Clean up
docker system prune -f

# Build with no cache to ensure fresh build
echo "Building Docker image..."
docker build -t 3dtopia-xl-shintaro:latest . --no-cache

echo "Build completed!"

# Test if GPU available
if command -v nvidia-smi &> /dev/null; then
    echo "Testing GPU access..."
    docker run --gpus all --rm 3dtopia-xl-shintaro:latest nvidia-smi
    
    echo "Testing robot generation..."
    docker run --gpus all --rm \
      -v $(pwd)/outputs:/workspace/outputs \
      3dtopia-xl-shintaro:latest \
      python enhanced_robot.py --prompt "a futuristic robot"
else
    echo "No GPU available for testing"
fi

echo "=== Ready for Lightning.ai! ==="