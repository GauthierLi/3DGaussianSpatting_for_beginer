"""
    combine train, validate, interactive visualization
"""
import os
import argparse
import subprocess

from typing import List

def install_requirements():
    def _excute(cmd: List[str], exec_dir: str = None):
        try:
            print(cmd)
            proc = subprocess.run(cmd,
                        stderr=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        text=True,
                        cwd=exec_dir,
                        bufsize=1)
            for line in proc.stdout.splitlines():
                print(line)

        except Exception as e:
            print(f"Error installing requirements: {e}")
    install_requirements_cmd = [
        "pip", "install", "-r", "requirements.txt"
    ]
    _excute(install_requirements_cmd)

    modules = os.listdir("submodules")
    for module in modules:
        exec_dir = os.path.join("submodules", module)
        module_install_cmd = ["python", "setup.py", "develop", "--user"]
        _excute(module_install_cmd, exec_dir=exec_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Train, validate, and visualize 3dGS models.")
    parser.add_argument("--install", action="store_true", default=False, help="Install requirements.")
    parser.add_argument("--mode", type=str, choices=["train", "validate", "visualize"], default="train", help="Operation mode.")
    return parser.parse_args()

def init():
    ...

def main(args: argparse.Namespace):
    if args.install:
        install_requirements()
    init()

if __name__ == "__main__":
    args = parse_args()
    main(args)
