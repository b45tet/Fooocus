"""Install and run Fooocus on the Colab GPU runtime (Python 3.13)."""

import os
from pathlib import Path
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parent


def run_step(label, command, **kwargs):
    print(f"[Fooocus Colab] {label}", flush=True)
    result = subprocess.run(command, **kwargs)
    if result.returncode:
        sys.exit(f"[Fooocus Colab] {label} failed (exit status {result.returncode}). See the output above.")


def main():
    if sys.version_info[:2] != (3, 13):
        raise SystemExit(
            f"Found Python {sys.version.split()[0]}. Select a Colab runtime with "
            "Python 3.13 and a GPU accelerator."
        )

    run_step("Check Python packages and CUDA", [sys.executable, "-u", "-c", """
import torch
import torchvision

print(f'PyTorch {torch.__version__}, torchvision {torchvision.__version__}, CUDA build: {torch.version.cuda}')
if tuple(int(part) for part in torch.__version__.split('+')[0].split('.')[:2]) < (2, 11):
    raise SystemExit('Fooocus needs PyTorch 2.11 or newer. Use a current Colab runtime.')
if not torch.cuda.is_available():
    raise SystemExit('PyTorch cannot access CUDA. Select a GPU accelerator in Colab (Runtime > Change runtime type) and reconnect.')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"""])

    env_dir = ROOT / ".venv-colab"
    python = env_dir / "bin" / "python"
    if not python.is_file():
        print("[Fooocus Colab] Create the virtual environment", flush=True)
        # Colab's Python may lack ensurepip; the system pip (via system site-packages) installs into the venv.
        venv.EnvBuilder(system_site_packages=True, with_pip=False).create(env_dir)

    run_step(
        "Install dependencies",
        [str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements_colab.txt")],
    )
    run_step(
        "Launch Fooocus",
        [str(python), "-u", str(ROOT / "launch.py"), *sys.argv[1:]],
        cwd=ROOT,
        env={**os.environ, "FOOOCUS_SKIP_INSTALL": "1", "USE_TF": "0", "USE_FLAX": "0"},
    )


if __name__ == "__main__":
    main()
