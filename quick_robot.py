#!/usr/bin/env python3
"""
Quick robot generation script for Lightning.ai
"""
import subprocess
import sys

def main():
    print("🤖 Generating a futuristic robot...")
    
    try:
        subprocess.run([
            "python", "/workspace/enhanced_robot.py",
            "--prompt", "a futuristic robot",
            "--output", "/workspace/outputs/robot",
            "--steps", "50",
            "--cfg", "5.0",
            "--seed", "42"
        ], check=True)
        
        print("✅ Robot generation completed!")
        print("📁 Check /workspace/outputs/robot/ for GLB files")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()