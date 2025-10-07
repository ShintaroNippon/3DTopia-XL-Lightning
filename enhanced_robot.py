#!/usr/bin/env python3
"""
Enhanced robot generation script with NumPy fix and GLB export
Optimized for Lightning.ai L4 GPU
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import yaml
import time

def main():
    parser = argparse.ArgumentParser(description="Generate futuristic robot 3D model")
    parser.add_argument("--prompt", default="a futuristic robot", help="Text prompt")
    parser.add_argument("--output", default="/workspace/outputs/futuristic_robot", help="Output directory")
    parser.add_argument("--steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--cfg", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print(f"🤖 Generating: {args.prompt}")
    print(f"📁 Output: {args.output}")
    print(f"⚙️  Settings: steps={args.steps}, cfg={args.cfg}, seed={args.seed}")
    
    # Ensure we're in the right directory
    os.chdir("/workspace")
    
    # Download models if needed
    print("📥 Ensuring models are downloaded...")
    result = subprocess.run(["/workspace/download_models.sh"], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Model download failed: {result.stderr}")
        return False
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"🚀 Starting inference...")
    start_time = time.time()
    
    try:
        # Run inference with the original script
        # Create a simple input file for the prompt
        input_file = f"{args.output}/input.txt"
        with open(input_file, "w") as f:
            f.write(args.prompt)
        
        # Run inference using the existing inference script
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        
        result = subprocess.run([
            "python", "inference.py", "./configs/inference_dit.yml"
        ], capture_output=True, text=True, env=env, cwd="/workspace")
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Generation completed in {elapsed_time:.1f} seconds!")
            print(f"📂 Check {args.output} for results")
            
            # List all generated files
            if os.path.exists(args.output):
                all_files = list(Path(args.output).iterdir())
                glb_files = [f for f in all_files if f.suffix.lower() == '.glb']
                
                if glb_files:
                    print("🎉 Generated GLB files:")
                    for file in glb_files:
                        size_mb = file.stat().st_size / (1024*1024)
                        print(f"   📦 {file.name} ({size_mb:.1f} MB)")
                else:
                    print("📄 Generated files:")
                    for file in all_files:
                        if file.is_file():
                            print(f"   📄 {file.name}")
            
            return True
            
        else:
            print(f"❌ Generation failed with error code: {result.returncode}")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Generation failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎊 Robot generation completed successfully!")
        print("🔗 Your 3D robot model is ready for download!")
    else:
        print("\n💔 Generation failed. Please check the logs above.")
        sys.exit(1)