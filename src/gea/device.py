# Device resolution for the Intel Arc B580 (PyTorch's "xpu" backend).
#
# Nothing in the stub backend touches the device. It matters from Day 3 of
# BUILD_PLAN.md, when ColQwen2 and SPECTER2 are loaded for real.
import warnings

import torch


def get_device() -> torch.device:
    # The B580 if PyTorch can see it. Otherwise the CPU, with a warning: the stub
    # sections still run, but the real models would be very slow.
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    warnings.warn("PyTorch cannot see the Arc B580 -- is the xpu build of torch installed "
                  "(BUILD_PLAN.md, Day 1)? Falling back to the CPU.")
    return torch.device("cpu")


def get_model_dtype(device: torch.device) -> torch.dtype:
    # bf16 on the B580 (its XMX engines run it natively, and it halves memory);
    # fp32 on the CPU.
    return torch.bfloat16 if device.type == "xpu" else torch.float32


def empty_cache(device: torch.device) -> None:
    # Hand cached GPU memory back -- call it after unloading a model.
    if device.type == "xpu":
        torch.xpu.empty_cache()


def describe_device(device: torch.device) -> str:
    if device.type == "xpu":
        props = torch.xpu.get_device_properties(0)
        return f"{props.name}, {props.total_memory / 2**30:.1f} GB"
    return "CPU"


def free_memory_gb() -> float:
    # Memory Windows can still hand out: free RAM plus free page file ("commit"). When
    # it runs out, ANY program that asks for more crashes -- including another
    # experiment running on this PC. Loading a model takes a lot of it at once.
    import ctypes
    if not hasattr(ctypes, "windll"):
        return float("inf")

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ("total_ram", "free_ram", "commit_limit", "free_commit",
             "total_virtual", "free_virtual", "free_extended")]

    status = MemoryStatus(dwLength=ctypes.sizeof(MemoryStatus))
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.free_commit / 2**30


def gpu_memory_in_use_mb():
    # Dedicated GPU memory in use by ALL programs, as Windows counts it (PyTorch only
    # sees its own). None if Windows cannot tell. Same check as scripts/start_llm.sh.
    import subprocess
    command = (r'[int](((Get-Counter "\GPU Adapter Memory(*)\Dedicated Usage").CounterSamples'
               r' | Measure-Object CookedValue -Maximum).Maximum / 1MB)')
    try:
        out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def room_problems(need_gb: float, gpu_limit_mb=None) -> list:
    # Why a model should NOT be loaded now, in plain words (empty list = go ahead):
    # too little free memory, or other programs holding more than gpu_limit_mb of the
    # B580. Checked before loading, because running out crashes everything running.
    problems = []
    free = free_memory_gb()
    if free < need_gb:
        problems.append(f"only {free:.1f} GB of memory is free; this needs about {need_gb} GB")
    if gpu_limit_mb is not None:
        used = gpu_memory_in_use_mb()
        if used is not None and used > gpu_limit_mb:
            problems.append(f"other programs hold {used:,} MB of the GPU; this needs it below {gpu_limit_mb:,} MB")
    return problems


def report_no_room(problems: list) -> None:
    for p in problems:
        print(f"Not enough room: {p}.")
    print("Close other programs (Task Manager shows which) or wait for them to finish, then run "
          "this again. --skip-checks starts anyway.")
