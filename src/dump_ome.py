#!/usr/bin/env python3
"""
dump_ome.py — Structural dump for Dark Earth .OME (Object Mesh Editor) files.

Purpose: help reverse-engineer the "3DITOR v1.3 / v2.0" mesh container so we
can port a native C loader in dke/src/model_ome.c. Not a decoder — this tool
just walks visible tags, reports offsets, and prints candidate integer /
float interpretations of the surrounding words.

Format overview (empirical, from bin/game/datas/L*/IM*.OME):
  +0x000  '3DITOR v1.3 MATR' + 16 dashes  # outer container = MATR (materials)
  +0x020  filename padded to 16 bytes ("IM01.MAT")
  +0x030  full path padded to 48 bytes    ("F:\\DEV\\DKE\\...")
  +0x060  MATR chunk body (per-material tags like CUIR, ACIER, ROUIL, ...)
  +0x???  '3DITOR v2.0 OBJC' + dashes     # nested sub-file = OBJC (object)
  +0x???  '3DITOR v2.0 ATOMF:...'         # nested sub-file = ATOMF (atoms/geometry)

Usage: python3 dump_ome.py path/to/IM01.OME [--offsets] [--dump-atomf]
"""

import argparse
import os
import re
import struct
import sys

SIG_RE = re.compile(rb"3DITOR v[0-9.]+ [A-Z]{3,6}")


def find_signatures(data: bytes):
    """Return list of (offset, full_header_bytes, tag_str) for every 3DITOR
    signature found in the file. Each signature marks the start of a nested
    chunk with its own filename+path header (see dump_header)."""
    hits = []
    for m in SIG_RE.finditer(data):
        header = m.group()
        # tag = last token, stripped of trailing space and terminator noise
        tag = header.split()[-1].decode("ascii", errors="replace").strip()
        hits.append((m.start(), header, tag))
    return hits


def dump_header(data: bytes, off: int, name_len: int = 16, path_len: int = 48):
    """Dump the fixed-layout header a 3DITOR chunk starts with: 16 bytes
    signature + 16 bytes dashes + name + path (both null-padded)."""
    sig_end = off + 16
    if sig_end + 16 > len(data):
        return
    dashes = data[sig_end : sig_end + 16]
    is_dash = all(b == 0x2D for b in dashes)
    name_off = sig_end + 16
    name = data[name_off : name_off + name_len].split(b"\x00", 1)[0]
    path_off = name_off + name_len
    path = data[path_off : path_off + path_len].split(b"\x00", 1)[0]
    print(
        f"  header: sig={data[off:sig_end]!r} dashes={'yes' if is_dash else dashes.hex()}"
    )
    print(f"          name={name!r} path={path!r}")


def scan_material_tags(data: bytes, base: int, end: int):
    """Between two signatures, log short uppercase tags (likely material
    ids: CUIR, CUIVR, ACIER, ROUIL, ...). Each tag is followed by a small
    binary record whose layout we don't fully know yet."""
    print(f"  scanning material tags in [0x{base:x} .. 0x{end:x}]:")
    for m in re.finditer(rb"[A-Z]{3,8}", data[base:end]):
        s = m.group().decode("ascii")
        off = base + m.start()
        surrounding = data[off - 4 : off + len(s) + 20]
        i32s = []
        for k in range(0, min(24, len(surrounding)), 4):
            if k + 4 <= len(surrounding):
                (v,) = struct.unpack_from("<i", surrounding, k)
                i32s.append(v)
        print(f"    0x{off:05x}: {s!r} bytes={surrounding.hex()} ints={i32s}")


def dump_atomf_records(data: bytes, base: int):
    """After the ATOMF header, a table of fixed-width records seems to
    follow at ~+0x70 offset (per IM01.OME visual scan). Each record ~32
    bytes with i32 fields. Print them so we can guess the mesh layout."""
    # find end of ASCII header (path terminator + name block)
    # heuristic: first repeating 32-byte pattern with a small u32 leader
    body = base + 0x70
    print(f"  ATOMF body probe from 0x{body:x}:")
    for i in range(body, min(body + 0x200, len(data)), 32):
        row = data[i : i + 32]
        if len(row) < 32:
            break
        f8 = struct.unpack("<8i", row)
        f4f = struct.unpack("<8f", row)
        print(
            f"    0x{i:05x}: i32={f8}\n"
            f"             f32={[f'{v:.3g}' for v in f4f]}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--dump-atomf", action="store_true",
                    help="Also dump candidate record rows after each ATOMF header.")
    ap.add_argument("--material-tags", action="store_true",
                    help="Scan for short uppercase tags between signatures.")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"file not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    with open(args.path, "rb") as f:
        data = f.read()
    print(f"file: {args.path}  size={len(data)}")

    hits = find_signatures(data)
    print(f"3DITOR signatures found: {len(hits)}")
    for i, (off, hdr, tag) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(data)
        print(f"\n[{i}] 0x{off:05x}  tag={tag!r}  span={end - off} bytes")
        dump_header(data, off)
        if args.material_tags and tag == "MATR":
            scan_material_tags(data, off + 0x60, end)
        if args.dump_atomf and tag.startswith("ATOM"):
            dump_atomf_records(data, off)


if __name__ == "__main__":
    main()
