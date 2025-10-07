#!/bin/bash
set -e

echo "🔧 Setting up fresh 3DTopia-XL repository..."

# Remove existing repo if it exists
rm -rf 3DTopia-XL-Lightning

# Create new directory
mkdir 3DTopia-XL-Lightning
cd 3DTopia-XL-Lightning

# Initialize git
git init
git branch -M main

# Create the files (you'll need to copy the content above)
echo "📝 Create these files with the content provided above:"
echo "   - Dockerfile"
echo "   - lightning.yaml"
echo "   - docker-build.sh"
echo "   - quick_robot.py"
echo "   - setup-repo.sh"

# Set up remote (replace with your actual repo URL)
git remote add origin https://github.com/ShintaroNippon/3DTopia-XL-Lightning.git

echo "✅ Repository structure ready!"
echo "📋 Next steps:"
echo "   1. Create the repository on GitHub: ShintaroNippon/3DTopia-XL-Lightning"
echo "   2. Copy all the file contents above into their respective files"
echo "   3. Run: git add . && git commit -m 'Initial commit' && git push -u origin main"
echo "   4. Import into Lightning.ai Studio"