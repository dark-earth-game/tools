#!/usr/bin/env python3
"""
Convert old PCX image files into a readable PNG.
Btw, if you read this. The people who made Dark Earth are INSANE.
Crazy structure, insane ways of mapping files....
Anyways. Just drop the PCX file onto this script and you're good.

Usage:
  python pcx_converter.py SC00039.PCX
  python pcx_converter.py SC00039.PCX -o SC00039.png
  python pcx_converter.py ./pcx_folder --out-dir ./converted

Requires:
  pip install pillow
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install it with: pip install pillow", file=sys.stderr)
    raise SystemExit(1)


def convert_pcx(input_path: Path, output_path: Path) -> None:
    """Convert one PCX file to PNG or another Pillow-supported image format."""
    try:
        with Image.open(input_path) as img:
            # Keep palette images readable, but convert unusual modes safely.
            # RGBA preserves transparency if it ever exists; RGB is best for old PCX files.
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)
            print(f"Converted: {input_path} -> {output_path}")
    except Exception as exc:
        print(f"Failed to convert {input_path}: {exc}", file=sys.stderr)


def iter_pcx_files(path: Path):
    if path.is_dir():
        yield from sorted(path.glob("*.pcx"))
        yield from sorted(path.glob("*.PCX"))
    else:
        yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert PCX files to readable PNG images.")
    parser.add_argument("input", help="A .PCX file or a folder containing .PCX files")
    parser.add_argument("-o", "--output", help="Output file path, only valid when input is one file")
    parser.add_argument("--out-dir", default="converted", help="Output folder when converting a directory")
    parser.add_argument("--format", default="png", choices=["png", "jpg", "bmp", "tiff"], help="Output image format")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    files = list(iter_pcx_files(input_path))
    if not files:
        print(f"No PCX files found in: {input_path}", file=sys.stderr)
        return 1

    if input_path.is_file() and args.output:
        output_path = Path(args.output)
        convert_pcx(input_path, output_path)
        return 0

    if input_path.is_file():
        output_path = input_path.with_suffix(f".{args.format}")
        convert_pcx(input_path, output_path)
        return 0

    out_dir = Path(args.out_dir)
    for pcx in files:
        output_path = out_dir / f"{pcx.stem}.{args.format}"
        convert_pcx(pcx, output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
