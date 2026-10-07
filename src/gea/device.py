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
