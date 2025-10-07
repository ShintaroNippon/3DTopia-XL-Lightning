#!/bin/bash

echo "Fixing Git setup for ShintaroNippon account..."

# Remove existing origin
git remote remove origin 2>/dev/null || true

# Add correct origin
git remote add origin https://github.com/ShintaroNippon/3DTopia-XL-Lightning.git

# Set up branch
git branch -M main

echo "Git setup complete. Now you can:"
echo "1. Make sure you're authenticated as ShintaroNippon"
echo "2. Run: git push -u origin main"

# Check current user
echo "Current git user: $(git config user.name)"
echo "Current git email: $(git config user.email)"

echo ""
echo "To change git credentials to ShintaroNippon:"
echo "git config --global user.name 'ShintaroNippon'"
echo "git config --global user.email 'your-shintaro-email@example.com'"