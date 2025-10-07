#!/bin/bash
set -e

echo "Setting up complete 3DTopia-XL repository..."

# Clone the original 3DTopia-XL repository content
if [ ! -d "3DTopia-XL-temp" ]; then
    echo "Cloning 3DTopia-XL repository..."
    git clone https://github.com/3DTopia/3DTopia-XL.git 3DTopia-XL-temp
fi

# Copy all necessary files (excluding .git)
echo "Copying files..."
cp -r 3DTopia-XL-temp/* ./
rm -rf 3DTopia-XL-temp

echo "Files copied successfully!"