"""Install and run Fooocus using the Colab 2026.10 GPU runtime."""

import os
from pathlib import Path
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parent


def main():
    if sys.version_info[:2] != (3, 13):
        raise SystemExit("Select the Colab 2026.10 runtime (Python 3.13) and a GPU accelerator.")

    subprocess.run([sys.executable, "-c", """
import torch
import torchvision

if (torch.__version__.split('+')[0], torchvision.__version__.split('+')[0]) != ('2.11.0', '0.26.0'):
    raise SystemExit('Select the Colab 2026.10 runtime with PyTorch 2.11.0 and torchvision 0.26.0.')
if not torch.cuda.is_available():
    raise SystemExit('Select a GPU accelerator in Colab before launching Fooocus.')
"""], check=True)

    env_dir = ROOT / ".venv-colab"
    python = env_dir / "bin" / "python"
    if not python.is_file():
        # Colab's Python may lack ensurepip; the system pip (via system site-packages) installs into the venv.
        venv.EnvBuilder(system_site_packages=True, with_pip=False).create(env_dir)

    subprocess.run(
        [str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements_colab.txt")],
        check=True,
    )
    subprocess.run(
        [str(python), str(ROOT / "launch.py"), *sys.argv[1:]],
        cwd=ROOT,
        env={**os.environ, "FOOOCUS_SKIP_INSTALL": "1", "USE_TF": "0", "USE_FLAX": "0"},
        check=True,
    )


if __name__ == "__main__":
    main()
