"""Test all AI-RAN dependencies."""

# 1. pyAerial
import aerial
print("[PASS] 1. aerial import OK")

from aerial.pycuphy import _pycuphy
n = len([x for x in dir(_pycuphy) if not x.startswith("_")])
print(f"[PASS] 2. _pycuphy C extension — {n} symbols")

# 3. PyTorch
import torch
cuda_ok = torch.cuda.is_available()
print(f"[PASS] 3. PyTorch {torch.__version__}, CUDA={cuda_ok}")
if cuda_ok:
    print(f"       GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB")

# 4. TensorFlow
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"[PASS] 4. TensorFlow {tf.__version__}, GPUs={len(gpus)}")

# 5. Sionna
import sionna
print(f"[PASS] 5. Sionna {sionna.__version__}")

# 6. CuPy
import cupy as cp
print(f"[PASS] 6. CuPy {cp.__version__}")

print()
print("=== ALL DEPENDENCIES READY ===")
