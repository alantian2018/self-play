"""Export a training checkpoint as a small, model-only release."""
import argparse
from pathlib import Path

from selfplay.checkpoints import export_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export_checkpoint(args.source, args.destination)
    print(f"Exported {args.destination} ({args.destination.stat().st_size / 1024:.1f} KiB)")
