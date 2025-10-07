#!/usr/bin/env python3
"""
Lightweight GPU diagnostics for this pipeline.

Checks:
- NVIDIA driver and device visibility (nvidia-smi)
- PyTorch CUDA (version, availability, simple matmul)

Exit code:
- 0 if all requested components either pass or are not installed (but no hard failures)
- 1 if any installed GPU-capable component fails its GPU test

Run:
  Windows PowerShell/cmd:
    python scripts\check_gpu_acceleration.py
  Linux/WSL:
    python3 scripts/check_gpu_acceleration.py
"""

from __future__ import annotations
import os
import sys
import platform
import subprocess
import shutil
import textwrap


def _print(title: str, body: str = "", ok: bool | None = None):
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        style = None
        if ok is True:
            style = "green"
        elif ok is False:
            style = "red"
        console = Console()
        console.print(Panel.fit(Text(body or "", style=style), title=title, border_style=style or "cyan"))
    except Exception:
        prefix = "[OK] " if ok is True else ("[FAIL] " if ok is False else "")
        print(f"\n{prefix}{title}\n{body}")


def check_nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        _print("nvidia-smi", "Not found on PATH. If on Windows, ensure NVIDIA driver installed. If in WSL, install GPU driver and use WSL2.", ok=False)
        return False, "nvidia-smi not found"
    try:
        out = subprocess.check_output([exe], stderr=subprocess.STDOUT, text=True, timeout=8)
        head = "\n".join(out.strip().splitlines()[:8])
        _print("nvidia-smi", head, ok=True)
        return True, None
    except Exception as e:
        _print("nvidia-smi", f"Execution failed: {e}", ok=False)
        return False, str(e)


def check_torch():
    try:
        import torch
    except Exception as e:
        _print("PyTorch", f"Not installed ({e}). Skipping Torch checks.")
        return None, None
    details = []
    details.append(f"Torch version: {getattr(torch, '__version__', 'unknown')}")
    details.append(f"Built with CUDA: {getattr(getattr(torch, 'version', None), 'cuda', None)}")
    avail = bool(torch.cuda.is_available())
    details.append(f"CUDA available: {avail}")
    if avail:
        try:
            name = torch.cuda.get_device_name(0)
            details.append(f"Device[0]: {name}")
            # Tiny GPU compute test
            a = torch.randn((1024, 512), device='cuda')
            b = torch.randn((512, 128), device='cuda')
            c = a @ b
            torch.cuda.synchronize()
            details.append("Matmul test: OK")
            _print("PyTorch CUDA", "\n".join(details), ok=True)
            return True, None
        except Exception as e:
            details.append(f"Matmul test failed: {e}")
            _print("PyTorch CUDA", "\n".join(details), ok=False)
            return False, str(e)
    else:
        _print("PyTorch CUDA", "\n".join(details), ok=False)
        return False, "torch.cuda.is_available() == False"


def summarize(results):
    lines = ["Environment:"]
    lines.append(f"  OS: {platform.system()} {platform.release()} | Python {platform.python_version()}")
    if os.environ.get('WSL_DISTRO_NAME'):
        lines.append(f"  WSL: {os.environ.get('WSL_DISTRO_NAME')}")
    lines.append("")
    for name, (status, err) in results.items():
        if status is True:
            lines.append(f"  [OK] {name}")
        elif status is False:
            lines.append(f"  [FAIL] {name}: {err}")
        else:
            lines.append(f"  [SKIP] {name} (not installed)")
    _print("Summary", "\n".join(lines))


def guidance():
    tips = []
    tips.append("If Torch CUDA is False on Windows: install CUDA-enabled torch wheel:")
    tips.append("  pip uninstall -y torch torchvision torchaudio")
    tips.append("  pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.0 torchvision torchaudio")
    tips.append("")
    tips.append("Ensure NVIDIA driver is installed; 'nvidia-smi' should work in the shell.")
    _print("Next Steps (if failures)", "\n".join(tips))


def main():
    results = {}
    results["nvidia-smi"] = check_nvidia_smi()
    results["PyTorch CUDA"] = check_torch()
    summarize(results)
    # Exit with 1 if a component is installed but failed
    failed = any(status is False for (status, _) in results.values())
    if failed:
        guidance()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

