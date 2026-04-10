"""HBM filler utility. Allocates GPU memory to a target occupancy percentage."""
import torch


def fill_hbm(target_percent):
    """Allocate GPU tensors to fill HBM to target_percent.

    Returns list of tensors (keep references alive to hold memory).
    """
    total = torch.cuda.get_device_properties(0).total_memory
    current_used = total - torch.cuda.mem_get_info()[0]
    target_bytes = int(total * target_percent / 100.0)
    to_alloc = target_bytes - current_used

    tensors = []
    if to_alloc > 0:
        # Allocate in 256MB chunks to avoid fragmentation issues
        chunk = 256 * 1024 * 1024
        while to_alloc > chunk:
            tensors.append(torch.zeros(chunk // 4, dtype=torch.float32, device="cuda"))
            to_alloc -= chunk
        if to_alloc > 0:
            tensors.append(torch.zeros(to_alloc // 4, dtype=torch.float32, device="cuda"))

    free, total = torch.cuda.mem_get_info()
    actual_percent = (total - free) / total * 100
    print(f"HBM: {(total-free)/1e9:.1f}/{total/1e9:.1f} GB ({actual_percent:.0f}%) "
          f"[target={target_percent}%]")
    return tensors


def get_hbm_info():
    """Return (used_bytes, total_bytes, percent)."""
    free, total = torch.cuda.mem_get_info()
    used = total - free
    return used, total, used / total * 100
