#!/bin/bash
set -e

echo "🔧 Setting up complete 3DTopia-XL repository..."

# Step 1: Get the original 3DTopia-XL code
if [ ! -f "inference.py" ]; then
    echo "📥 Cloning 3DTopia-XL repository..."
    git clone https://github.com/3DTopia/3DTopia-XL.git temp-repo
    
    # Copy all files except .git
    cp -r temp-repo/* ./
    rm -rf temp-repo
    
    echo "✅ Repository code copied!"
fi

# Step 2: Create directories
mkdir -p pretrained inputs outputs logs

# Step 3: Commit and push everything
echo "📤 Pushing to GitHub..."
git add .
git commit -m "Add complete 3DTopia-XL code with NumPy fix and L4 optimization"
git push origin main

echo "✅ Repository is ready!"
echo "🚀 Next: Import this repository into Lightning.ai Studio"
echo ""
echo "Repository URL: https://github.com/ShintaroNippon/3DTopia-XL-Lightning"