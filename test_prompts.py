#!/usr/bin/env python3
"""
Quick test script for generating different 3D models
Usage: python test_prompts.py
"""

import subprocess
import sys
import time
from pathlib import Path

# Test prompts
PROMPTS = [
    "a futuristic robot",
    "a vintage sports car", 
    "a medieval castle tower",
    "a modern coffee mug",
    "a sleek spaceship"
]

def test_generation(prompt):
    """Test generation for a single prompt"""
    print(f"\n{'='*50}")
    print(f"🎯 Testing: {prompt}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        # Run generation
        result = subprocess.run([
            "python", "/workspace/generate_3d.py", 
            "--prompt", prompt,
            "--output", "/workspace/outputs"
        ], capture_output=True, text=True, cwd="/workspace")
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS in {duration:.1f}s")
            print(result.stdout)
            
            # Check for GLB files
            output_dir = Path("/workspace/outputs") / prompt.replace(" ", "_")[:50]
            glb_files = list(output_dir.glob("*.glb"))
            if glb_files:
                print(f"📦 Generated {len(glb_files)} GLB file(s):")
                for file in glb_files:
                    print(f"   📄 {file.name} ({file.stat().st_size / 1024 / 1024:.1f} MB)")
            
        else:
            print(f"❌ FAILED in {duration:.1f}s")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False

def main():
    """Run tests for all prompts"""
    print("🧪 Starting 3DTopia-XL Prompt Tests")
    print("=" * 60)
    
    results = []
    total_start = time.time()
    
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n🔄 Test {i}/{len(PROMPTS)}")
        success = test_generation(prompt)
        results.append((prompt, success))
        
        # Small delay between tests
        if i < len(PROMPTS):
            time.sleep(2)
    
    # Summary
    total_time = time.time() - total_start
    successful = sum(1 for _, success in results if success)
    
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}/{len(PROMPTS)}")
    print(f"⏱️ Total time: {total_time:.1f}s")
    print(f"📁 Results in: /workspace/outputs/")
    
    print(f"\n📋 Detailed Results:")
    for prompt, success in results:
        status = "✅" if success else "❌"
        print(f"   {status} {prompt}")
    
    # Check outputs directory
    outputs_dir = Path("/workspace/outputs")
    if outputs_dir.exists():
        total_files = len(list(outputs_dir.rglob("*.glb")))
        print(f"\n📦 Total GLB files generated: {total_files}")

if __name__ == "__main__":
    main()