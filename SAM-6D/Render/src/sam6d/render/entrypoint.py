import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scriptname", choices=("render_custom_templates",))
    args, unknown = parser.parse_known_args()

    _dir = Path(__file__).parent.resolve()
    script = _dir.joinpath(args.scriptname).with_suffix(".py")

    cmd = ["blenderproc", "run", str(script)] + unknown
    subprocess.run(cmd, check=True)
