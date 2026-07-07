#!/usr/bin/env python3
"""
dump_lieu.py — extract data sections + exports from Dark Earth LIEU%02d.DLL
files into per-level binary blobs. Enables the C port to consume level data
without a PE loader.

Install:  pip install pefile

Usage:
    dump_lieu.py <file.dll> [<output-dir>]     # dump one file to output-dir (default ./out)
    dump_lieu.py --list <file.dll>             # print sections/exports/imports/strings only
    dump_lieu.py --all [<game-dir>]            # process every .DLL in game-dir (default bin/game/)
                                               # writing to dke/level_data/<stem>/
"""
import os
import re
import sys
from pathlib import Path

try:
    import pefile
except ImportError:
    print("[fatal] pefile not installed. Run: pip install pefile", file=sys.stderr)
    sys.exit(1)

# Sections we want to dump as raw bytes.
DUMP_SECTIONS = (".text", ".data", ".rdata", ".bss", ".rsrc")


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def print_header(pe, path):
    print(f"== {path} ==")
    print(f"  Machine        : 0x{pe.FILE_HEADER.Machine:04X}")
    print(f"  ImageBase      : 0x{pe.OPTIONAL_HEADER.ImageBase:08X}")
    print(f"  EP RVA         : 0x{pe.OPTIONAL_HEADER.AddressOfEntryPoint:08X}")
    print(f"  Characteristics: 0x{pe.FILE_HEADER.Characteristics:04X}")
    print(f"  Sections       : {pe.FILE_HEADER.NumberOfSections}")
    for s in pe.sections:
        name = s.Name.rstrip(b"\x00").decode("ascii", errors="replace")
        print(f"    {name:<8}  VA=0x{s.VirtualAddress:08X} "
              f"VSz=0x{s.Misc_VirtualSize:08X} "
              f"RawSz=0x{s.SizeOfRawData:08X} "
              f"Flags=0x{s.Characteristics:08X}")


def print_exports(pe):
    if not hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
        print("  No export table.")
        return []
    exp = pe.DIRECTORY_ENTRY_EXPORT
    dll_name = exp.name.decode("ascii", errors="replace") if exp.name else "?"
    print(f"  Export DLL name: {dll_name}")
    entries = []
    for e in exp.symbols:
        n = e.name.decode("ascii", errors="replace") if e.name else f"@{e.ordinal}"
        print(f"    ord={e.ordinal:<5} rva=0x{e.address:08X}  {n}"
              + (f"  -> {e.forwarder.decode('ascii', errors='replace')}"
                 if e.forwarder else ""))
        entries.append((n, e.address, e.ordinal))
    return entries


def print_imports(pe):
    if not hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        print("  No import table.")
        return
    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        dll = entry.dll.decode("ascii", errors="replace")
        print(f"  Import DLL: {dll}")
        for imp in entry.imports:
            n = imp.name.decode("ascii", errors="replace") if imp.name else f"@{imp.ordinal}"
            print(f"    0x{imp.address:08X}  {n}")


def extract_strings(pe, min_len=4):
    """Yield printable-ASCII strings from .rdata / .data sections."""
    for s in pe.sections:
        name = s.Name.rstrip(b"\x00").decode("ascii", errors="replace")
        if name not in (".rdata", ".data"):
            continue
        data = s.get_data()
        cur = bytearray()
        for b in data:
            if 0x20 <= b < 0x7F:
                cur.append(b)
            else:
                if len(cur) >= min_len:
                    yield name, cur.decode("ascii")
                cur.clear()
        if len(cur) >= min_len:
            yield name, cur.decode("ascii")


def print_strings(pe, limit=200):
    print("  Strings (>=4 chars, first 200):")
    for i, (sec, s) in enumerate(extract_strings(pe)):
        if i >= limit:
            print("    ... (truncated)")
            break
        print(f"    [{sec}] {s}")


def dump_sections(pe, out_dir, stem):
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in pe.sections:
        name = s.Name.rstrip(b"\x00").decode("ascii", errors="replace")
        if name not in DUMP_SECTIONS:
            continue
        # Use VirtualSize for accurate section size (rounded up by SizeOfRawData).
        vsize = s.Misc_VirtualSize
        try:
            data = s.get_data(length=vsize)
        except Exception as e:
            print(f"  [warn] section {name}: {e}", file=sys.stderr)
            continue
        out = out_dir / f"{stem}_{sanitize(name.lstrip('.'))}.bin"
        out.write_bytes(data)
        print(f"  Dumped {name}: {vsize} bytes -> {out}")


def dump_exports(pe, out_dir, stem, entries, max_bytes=4096):
    if not entries:
        return
    # Sort by RVA to compute per-export slice lengths.
    sorted_e = sorted(entries, key=lambda t: t[1])
    for i, (name, rva, ordinal) in enumerate(sorted_e):
        next_rva = sorted_e[i + 1][1] if i + 1 < len(sorted_e) else rva + max_bytes
        length = min(max(0, next_rva - rva), max_bytes)
        if length <= 0:
            continue
        try:
            data = pe.get_data(rva, length)
        except Exception as e:
            print(f"  [warn] export {name}: {e}", file=sys.stderr)
            continue
        out = out_dir / f"{stem}_export_{sanitize(name)}.bin"
        out.write_bytes(data)
        print(f"  Dumped export {name}: {length} bytes -> {out}")


def process_one(path, out_dir, list_only=False):
    try:
        pe = pefile.PE(str(path))
    except pefile.PEFormatError as e:
        print(f"[error] {path}: {e}", file=sys.stderr)
        return False

    print_header(pe, str(path))
    entries = print_exports(pe)
    print_imports(pe)
    print_strings(pe)

    if not list_only:
        stem = Path(path).stem
        dump_sections(pe, out_dir, stem)
        dump_exports(pe, out_dir, stem, entries)

    pe.close()
    return True


def main(argv):
    args = argv[1:]
    if not args:
        print(__doc__.strip())
        return 1

    if args[0] == "--list":
        if len(args) != 2:
            print(__doc__.strip())
            return 1
        return 0 if process_one(args[1], out_dir=None, list_only=True) else 1

    if args[0] == "--all":
        game_dir = Path(args[1]) if len(args) >= 2 else Path("bin/game")
        out_root = Path("dke/level_data")
        ok = True
        for dll in sorted(game_dir.glob("*.DLL")) + sorted(game_dir.glob("*.dll")):
            stem = dll.stem
            out_dir = out_root / stem
            print(f"\n### {dll.name} -> {out_dir}")
            if not process_one(dll, out_dir):
                ok = False
        return 0 if ok else 1

    # Default: one DLL, one output dir.
    path = args[0]
    out_dir = Path(args[1]) if len(args) >= 2 else Path("out")
    return 0 if process_one(path, out_dir) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
