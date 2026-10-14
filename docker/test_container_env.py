#!/usr/bin/env python3
"""
Comprehensive container environment smoke test script for Cover RL & DexMG.
Verifies CUDA, PyTorch, TorchRL, TensorDict, MuJoCo EGL headless rendering,
Robosuite, and LeRobot imports.
"""

import os
import sys

def test_cuda_and_torch():
    print("[1/5] Testing PyTorch & CUDA...")
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    if not torch.cuda.is_available():
        print("  [FAIL] CUDA is NOT available to PyTorch!")
        return False
    
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"  [PASS] Found {device_count} GPU(s). Device 0: {device_name} (Compute Capability: {capability[0]}.{capability[1]})")
    
    # Test tensor allocation and computation on GPU
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.matmul(x, x)
    assert y.shape == (1000, 1000)
    print("  [PASS] Basic GPU tensor computation succeeded.")
    return True

def test_mujoco_egl():
    print("\n[2/5] Testing MuJoCo Headless EGL Rendering...")
    os.environ["MUJOCO_GL"] = "egl"
    import mujoco
    print(f"  MuJoCo version: {mujoco.__version__}")
    
    xml = """
    <mujoco>
      <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="1 1 0.1"/>
        <body pos="0 0 1">
          <joint type="free"/>
          <geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=128, width=128)
    
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    pixels = renderer.render()
    
    assert pixels.shape == (128, 128, 3)
    assert pixels.mean() > 0
    print(f"  [PASS] Rendered offscreen frame successfully: shape={pixels.shape}, mean_pixel={pixels.mean():.2f}")
    return True

def test_torchrl_and_tensordict():
    print("\n[3/5] Testing TorchRL & TensorDict...")
    import tensordict
    import torchrl
    from tensordict import TensorDict
    import torch
    
    print(f"  TorchRL version: {torchrl.__version__}")
    print(f"  TensorDict version: {tensordict.__version__}")
    
    td = TensorDict({"a": torch.randn(10, 5, device="cuda")}, batch_size=[10])
    assert td["a"].device.type == "cuda"
    print("  [PASS] TensorDict GPU allocation succeeded.")
    return True

def test_robosuite():
    print("\n[4/5] Testing Robosuite & DexMimicGen...")
    import robosuite
    import dexmimicgen
    import mimicgen
    print(f"  Robosuite version: {robosuite.__version__}")
    print(f"  DexMimicGen loaded successfully.")
    print(f"  MimicGen loaded successfully.")
    print("  [PASS] Robosuite & DexMimicGen imports verified.")
    return True

def test_lerobot():
    print("\n[5/5] Testing LeRobot & Hugging Face...")
    import lerobot
    from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
    print(f"  LeRobot version: {lerobot.__version__}")
    print("  [PASS] LeRobot and DiffusionPolicy loader imported successfully.")
    return True

def main():
    print("=" * 60)
    print("Cover RL Container Environment Verification Test")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version   : {sys.version.split()[0]}")
    print("=" * 60)
    
    tests = [
        test_cuda_and_torch,
        test_mujoco_egl,
        test_torchrl_and_tensordict,
        test_robosuite,
        test_lerobot,
    ]
    
    all_passed = True
    for test in tests:
        try:
            passed = test()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL VERIFICATION CHECKS PASSED! Container is ready for training.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("SOME CHECKS FAILED. Please review the errors above.")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
