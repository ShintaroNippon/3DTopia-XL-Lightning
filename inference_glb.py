#!/usr/bin/env python3
"""
GLB Inference Script for 3DTopia-XL
Usage: python inference_glb.py --prompt "your prompt" --output_dir /path/to/output
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import yaml
import subprocess

def download_models():
    """Download required models"""
    print("📥 Downloading models...")
    os.makedirs("/workspace/pretrained", exist_ok=True)
    
    models = {
        "model_sview_dit_fp16.pt": "https://huggingface.co/FrozenBurning/3DTopia-XL/resolve/main/model_sview_dit_fp16.pt",
        "model_vae_fp16.pt": "https://huggingface.co/FrozenBurning/3DTopia-XL/resolve/main/model_vae_fp16.pt"
    }
    
    for model_name, url in models.items():
        model_path = f"/workspace/pretrained/{model_name}"
        if not os.path.exists(model_path):
            print(f"Downloading {model_name}...")
            os.system(f"wget -q -O {model_path} {url}")
    
    print("✅ Models downloaded!")

def create_config(prompt, output_dir, steps=50, cfg=5.0, seed=42):
    """Create inference configuration"""
    config = {
        "model": {
            "target": "models.dit_models.DiT3D_Latte",
            "params": {
                "input_size": 64,
                "patch_size": 2,
                "in_channels": 16,
                "hidden_size": 1152,
                "depth": 28,
                "num_heads": 16,
                "mlp_ratio": 4.0,
                "class_dropout_prob": 0.1,
                "num_classes": 1000,
                "learn_sigma": True
            }
        },
        "vae": {
            "target": "models.vae_models.AutoencoderKL",
            "params": {
                "embed_dim": 16
            }
        },
        "inference": {
            "prompt": prompt,
            "output_dir": output_dir,
            "ddim_steps": steps,
            "cfg_scale": cfg,
            "seed": seed,
            "export_glb": True,
            "export_mesh": True,
            "mesh_resolution": 256,
            "decimate_faces": 50000,
            "fast_unwrap": False,
            "remesh": False
        },
        "paths": {
            "dit_checkpoint": "/workspace/pretrained/model_sview_dit_fp16.pt",
            "vae_checkpoint": "/workspace/pretrained/model_vae_fp16.pt"
        }
    }
    
    return config

def run_inference(prompt, output_dir="/app/outputs", steps=50, cfg=5.0, seed=42):
    """Run 3DTopia-XL inference with GLB export"""
    
    print(f"🚀 Starting inference for: '{prompt}'")
    print(f"📁 Output directory: {output_dir}")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    # Download models
    download_models()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create safe filename from prompt
    safe_name = "".join(c for c in prompt if c.isalnum() or c in " -_").strip()
    safe_name = safe_name.replace(" ", "_")[:50]
    
    prompt_output_dir = os.path.join(output_dir, safe_name)
    os.makedirs(prompt_output_dir, exist_ok=True)
    
    # Create config
    config = create_config(prompt, prompt_output_dir, steps, cfg, seed)
    config_path = os.path.join(prompt_output_dir, "config.yml")
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"🔧 Config saved: {config_path}")
    
    # Run inference using the original script
    try:
        os.chdir("/workspace")
        
        # Set environment variables
        env = os.environ.copy()
        env["PROMPT"] = prompt
        env["OUTPUT_DIR"] = prompt_output_dir
        env["EXPORT_GLB"] = "true"
        env["PYTHONPATH"] = "/workspace"
        
        # Run the inference
        cmd = ["python3", "inference.py", config_path]
        
        print(f"🎯 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Inference completed successfully!")
            
            # List generated files
            output_files = list(Path(prompt_output_dir).glob("*"))
            if output_files:
                print(f"📦 Generated {len(output_files)} files:")
                for file in output_files:
                    if file.is_file():
                        size_mb = file.stat().st_size / (1024*1024)
                        print(f"   📄 {file.name} ({size_mb:.1f} MB)")
            
            # Look for GLB files specifically
            glb_files = list(Path(prompt_output_dir).glob("*.glb"))
            if glb_files:
                print(f"\n🎊 GLB files ready:")
                for glb in glb_files:
                    print(f"   🔗 {glb}")
            else:
                print("\n⚠️  No GLB files found. Checking for other 3D formats...")
                mesh_files = list(Path(prompt_output_dir).glob("*.obj")) + \
                           list(Path(prompt_output_dir).glob("*.ply")) + \
                           list(Path(prompt_output_dir).glob("*.mesh"))
                for mesh in mesh_files:
                    print(f"   📐 {mesh}")
            
            return True
            
        else:
            print(f"❌ Inference failed with return code: {result.returncode}")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"💥 Error during inference: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="3DTopia-XL GLB Generation")
    parser.add_argument("--prompt", required=True, help="Text prompt for 3D generation")
    parser.add_argument("--output_dir", default="/app/outputs", help="Output directory")
    parser.add_argument("--steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--cfg", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    success = run_inference(
        prompt=args.prompt,
        output_dir=args.output_dir,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed
    )
    
    if success:
        print(f"\n🎉 Successfully generated 3D model for: '{args.prompt}'")
        print(f"📂 Check {args.output_dir} for GLB files!")
    else:
        print(f"\n💔 Generation failed for: '{args.prompt}'")
        sys.exit(1)

if __name__ == "__main__":
    main()