#!/usr/bin/env python3
"""
lieu_extract.py — Extract call trace from LIEU%02d.DLL FonctionEditor@16 init.

Loads DLL at ImageBase (0x10000000) via unicorn, wires 40 fake engine callbacks
that trap when invoked, calls FonctionEditor(opcode=1, state, cb_table, flag=0),
and records every callback (name + arg registers + string args) as a script.

Output: `LIEU%02d.script` (text). One line per callback call:
    CB_<off> ecx=<hex> edx=<hex> str_ecx=<repr> str_edx=<repr>

Usage:
    python3 lieu_extract.py path/to/LIEU01.DLL [--out LIEU01.script]
"""

import argparse
import struct
import sys
from pathlib import Path

try:
    from unicorn import Uc, UC_ARCH_X86, UC_MODE_32, UC_HOOK_CODE, UC_HOOK_MEM_INVALID
    from unicorn.x86_const import (
        UC_X86_REG_ECX, UC_X86_REG_EDX, UC_X86_REG_EAX,
        UC_X86_REG_ESP, UC_X86_REG_EIP, UC_X86_REG_EBP,
    )
except ImportError:
    print("ERR: pip3 install --user unicorn", file=sys.stderr)
    sys.exit(1)

# Callback stack-pop table (retn N) per engine callback slot.
# Reconstructed by scanning dkev.exe near each callback's end.
CB_POPS = {
    0x00: 0,  0x04: 0,  0x08: 0,  0x0C: 0,  0x10: 0,
    0x14: 24, 0x18: 28, 0x1C: 12, 0x20: 12, 0x24: 0,
    0x28: 0,  0x2C: 4,  0x30: 16, 0x34: 0,  0x38: 0,
    0x3C: 0,  0x40: 0,  0x44: 0,  0x48: 0,  0x4C: 12,
    0x50: 0,  0x54: 0,  0x58: 0,  0x5C: 0,  0x60: 0,
    0x64: 4,  0x68: 0,  0x6C: 0,  0x70: 0,  0x74: 0,
    0x78: 0,  0x7C: 0,  0x80: 8,  0x84: 16, 0x88: 4,
    0x8C: 0,  0x90: 0,  0x94: 0,  0x98: 4,  0x9C: 4,
}

# Symbolic names for identified engine callbacks (from IDA analysis).
CB_NAMES = {
    0x00: "NamedObject",           # sub_44453E, action_id=0
    0x04: "NamedObjectFlag",       # sub_4449A5, action_id=1
    0x08: "NamedObjectTwoInts",    # sub_444A3E, action_id=2
    0x0C: "Cinematique",           # sub_444AC0, action_id=3
    0x10: "KillPerso",             # sub_444FF5
    0x14: "NamedObjectStrPair",    # sub_44502D, action_id=5
    0x18: "NamedObjectStr",        # sub_445189
}

# Memory layout (per-run).
IMAGE_BASE  = 0x10000000
CBT_ADDR    = 0x20000000    # fake engine callback table
TRAMP_ADDR  = 0x21000000    # trampoline stubs (one per callback)
TRAMP_SIZE  = 0x10          # bytes per trampoline
STATE_ADDR  = 0x30000000    # engine state buffer (0xFA0 bytes)
STACK_TOP   = 0x40100000    # esp initial value grows down toward 0x40000000
STACK_BASE  = 0x40000000
SENTINEL    = 0x50000000    # return-address halt

STATE_SIZE  = 0x1000
STACK_SIZE  = 0x100000

DLL_ALLOC   = 0x100000       # room for PE image (all sections)


def parse_pe(data):
    e_lfanew = struct.unpack('<I', data[0x3C:0x40])[0]
    n_secs   = struct.unpack('<H', data[e_lfanew + 6:e_lfanew + 8])[0]
    opt_sz   = struct.unpack('<H', data[e_lfanew + 20:e_lfanew + 22])[0]
    opt      = e_lfanew + 24
    image_base = struct.unpack('<I', data[opt + 28:opt + 32])[0]
    ep_rva     = struct.unpack('<I', data[opt + 16:opt + 20])[0]
    secs = []
    for i in range(n_secs):
        so = opt + opt_sz + i * 40
        name  = data[so:so + 8].rstrip(b'\x00').decode('ascii', 'replace')
        vsize = struct.unpack('<I', data[so + 8:so + 12])[0]
        vaddr = struct.unpack('<I', data[so + 12:so + 16])[0]
        rsize = struct.unpack('<I', data[so + 16:so + 20])[0]
        roff  = struct.unpack('<I', data[so + 20:so + 24])[0]
        secs.append((name, vaddr, vsize, roff, rsize))
    # Export dir @ opt + 96 (DataDirectory[0])
    exp_rva = struct.unpack('<I', data[opt + 96:opt + 100])[0]
    return image_base, secs, exp_rva


def map_image(uc, data, secs):
    # Align to 0x1000 pages, allocate DLL_ALLOC total
    uc.mem_map(IMAGE_BASE, DLL_ALLOC)
    # Write PE headers (first 0x1000 bytes)
    uc.mem_write(IMAGE_BASE, data[:min(0x1000, len(data))])
    # Write each section at its VA
    for name, vaddr, vsize, roff, rsize in secs:
        chunk = data[roff:roff + rsize]
        uc.mem_write(IMAGE_BASE + vaddr, chunk)


def resolve_export(data, image_base, exp_rva, secs, name_wanted):
    """Return VA of export by name."""
    # ExportDirectory layout: +0x1c ptr AddressOfFunctions, +0x20 ptr AddressOfNames,
    # +0x24 ptr AddressOfNameOrdinals, +0x14 NumberOfFunctions, +0x18 NumberOfNames
    def rva_to_off(rva):
        for _, vaddr, vsize, roff, rsize in secs:
            if vaddr <= rva < vaddr + vsize:
                return roff + (rva - vaddr)
        return None
    exp_off = rva_to_off(exp_rva)
    if exp_off is None or exp_off + 0x28 > len(data):
        return None
    n_names = struct.unpack('<I', data[exp_off + 0x18:exp_off + 0x1C])[0]
    fn_tbl  = struct.unpack('<I', data[exp_off + 0x1C:exp_off + 0x20])[0]
    nm_tbl  = struct.unpack('<I', data[exp_off + 0x20:exp_off + 0x24])[0]
    ord_tbl = struct.unpack('<I', data[exp_off + 0x24:exp_off + 0x28])[0]
    fn_off  = rva_to_off(fn_tbl)
    nm_off  = rva_to_off(nm_tbl)
    ord_off = rva_to_off(ord_tbl)
    for i in range(n_names):
        name_rva = struct.unpack('<I', data[nm_off + i * 4:nm_off + i * 4 + 4])[0]
        name_off = rva_to_off(name_rva)
        end = data.index(b'\x00', name_off)
        nm = data[name_off:end].decode('ascii', 'replace')
        if nm == name_wanted:
            ordinal = struct.unpack('<H', data[ord_off + i * 2:ord_off + i * 2 + 2])[0]
            fn_rva  = struct.unpack('<I', data[fn_off + ordinal * 4:fn_off + ordinal * 4 + 4])[0]
            return image_base + fn_rva
    return None


def read_cstr(uc, addr, maxlen=128):
    if not addr or addr < 0x10000000 or addr > 0x50000000:
        return None
    try:
        buf = uc.mem_read(addr, maxlen)
    except Exception:
        return None
    end = bytes(buf).find(b'\x00')
    if end < 0:
        return None
    try:
        return bytes(buf[:end]).decode('ascii', 'replace')
    except Exception:
        return None


def looks_like_string(uc, addr):
    """Heuristic: is `addr` a valid ASCII c-string in the DLL image?"""
    if not addr:
        return False
    if not (IMAGE_BASE <= addr < IMAGE_BASE + DLL_ALLOC):
        return False
    s = read_cstr(uc, addr, 32)
    if not s or len(s) < 2:
        return False
    return all(0x20 <= ord(c) < 0x7f for c in s)


class Trace:
    def __init__(self):
        self.records = []
        self.stopped = False
        self.step_count = 0
        self.limit = 20_000_000

    def hook_code(self, uc, address, size, user):
        self.step_count += 1
        if self.step_count > self.limit:
            print(f"! step limit reached at 0x{address:x}", file=sys.stderr)
            uc.emu_stop()
            return

        if TRAMP_ADDR <= address < TRAMP_ADDR + 40 * TRAMP_SIZE:
            slot = (address - TRAMP_ADDR) // TRAMP_SIZE
            off  = slot * 4
            ecx  = uc.reg_read(UC_X86_REG_ECX)
            edx  = uc.reg_read(UC_X86_REG_EDX)
            esp  = uc.reg_read(UC_X86_REG_ESP)
            # Read stack args
            stack_args = []
            try:
                sb = uc.mem_read(esp + 4, 32)  # [esp+0] = return addr, [esp+4]... args
                for k in range(0, 32, 4):
                    stack_args.append(struct.unpack('<I', sb[k:k + 4])[0])
            except Exception:
                pass
            ret_addr = struct.unpack('<I', uc.mem_read(esp, 4))[0]

            name = CB_NAMES.get(off, f"CB_{off:02X}")
            pops = CB_POPS.get(off, 0)
            rec = {
                'slot': off, 'name': name,
                'ecx': ecx, 'edx': edx,
                'str_ecx': read_cstr(uc, ecx),
                'str_edx': read_cstr(uc, edx),
                'stack': stack_args[:4],
                'stack_strs': [read_cstr(uc, a) if looks_like_string(uc, a) else None
                              for a in stack_args[:4]],
                'pops': pops,
                'ret_addr': ret_addr,
            }
            self.records.append(rec)
            # Simulate stdcall: pop return addr, adjust esp by pops
            uc.reg_write(UC_X86_REG_ESP, esp + 4 + pops)
            uc.reg_write(UC_X86_REG_EAX, 0)
            uc.reg_write(UC_X86_REG_EIP, ret_addr)
            return

        if address == SENTINEL:
            self.stopped = True
            uc.emu_stop()
            return

    def hook_mem_invalid(self, uc, access, address, size, value, user):
        eip = uc.reg_read(UC_X86_REG_EIP)
        print(f"! invalid mem access {access} at 0x{address:x} (eip=0x{eip:x}, "
              f"size={size} val={value})", file=sys.stderr)
        return False   # don't recover; stop


def run(dll_path, out_path):
    data = Path(dll_path).read_bytes()
    image_base, secs, exp_rva = parse_pe(data)
    assert image_base == IMAGE_BASE, f"image base mismatch: {image_base:x}"

    fe_va = resolve_export(data, image_base, exp_rva, secs, "@FonctionEditor@16")
    if fe_va is None:
        print("ERR: @FonctionEditor@16 export not found", file=sys.stderr)
        return 1
    print(f"@FonctionEditor@16 @ 0x{fe_va:x}")

    uc = Uc(UC_ARCH_X86, UC_MODE_32)
    map_image(uc, data, secs)

    # Callback table + trampolines
    uc.mem_map(CBT_ADDR,   0x1000)      # 40 × 4 bytes = 160 bytes
    uc.mem_map(TRAMP_ADDR, 0x1000)      # 40 × TRAMP_SIZE bytes
    for i in range(40):
        off = i * 4
        uc.mem_write(CBT_ADDR + off, struct.pack('<I', TRAMP_ADDR + i * TRAMP_SIZE))
        # Trampoline body doesn't matter — hook intercepts at the trampoline VA.
        # Fill with 0xCC (int3) as safety net.
        uc.mem_write(TRAMP_ADDR + i * TRAMP_SIZE, b'\xCC' * TRAMP_SIZE)

    # State buffer
    uc.mem_map(STATE_ADDR, STATE_SIZE)
    # State buffer at [state+0] is later dereferenced in dispatch fallthrough;
    # set state[0] = state ptr itself so `mov eax, [state]; mov eax, [eax]` doesn't fault
    uc.mem_write(STATE_ADDR, struct.pack('<I', STATE_ADDR))

    # Stack
    uc.mem_map(STACK_BASE, STACK_SIZE)
    # Sentinel page (return address halt)
    uc.mem_map(SENTINEL & ~0xFFF, 0x1000)

    # Stack frame for FonctionEditor@16(ecx=1, edx=state, arg0=cbt, arg4=flag)
    # stdcall: caller pushes args right-to-left. edx/ecx via regs (thiscall vs stdcall
    # — the DLL entry does `mov [ebp-8], edx; mov [ebp-4], ecx`, so ecx/edx are used).
    # Stack layout: [ret_addr, arg0, arg4] where FonctionEditor is retn 8
    # -> caller-view: push flag; push cbt; call fe (implicit ret_addr push)
    esp = STACK_TOP
    esp -= 4; uc.mem_write(esp, struct.pack('<I', 0))            # arg4: reload_flag = 0
    esp -= 4; uc.mem_write(esp, struct.pack('<I', CBT_ADDR))      # arg0: callback table
    esp -= 4; uc.mem_write(esp, struct.pack('<I', SENTINEL))      # ret addr → halt
    uc.reg_write(UC_X86_REG_ESP, esp)
    uc.reg_write(UC_X86_REG_EBP, 0)
    uc.reg_write(UC_X86_REG_ECX, 1)                # opcode = 1 (init)
    uc.reg_write(UC_X86_REG_EDX, STATE_ADDR)       # state buffer

    trace = Trace()
    uc.hook_add(UC_HOOK_CODE, trace.hook_code)
    uc.hook_add(UC_HOOK_MEM_INVALID, trace.hook_mem_invalid)

    print(f"starting emulation at 0x{fe_va:x} ...")
    try:
        uc.emu_start(fe_va, SENTINEL, timeout=0, count=0)
    except Exception as e:
        eip = uc.reg_read(UC_X86_REG_EIP)
        print(f"! emulation error at eip=0x{eip:x}: {e}", file=sys.stderr)

    print(f"stopped after {trace.step_count} instructions, "
          f"{len(trace.records)} callback records")

    # Emit script
    lines = []
    for r in trace.records:
        parts = [f"{r['name']}({r['slot']:02X})"]
        if r['str_ecx'] is not None:
            parts.append(f"ecx={r['str_ecx']!r}")
        else:
            parts.append(f"ecx=0x{r['ecx']:x}")
        if r['str_edx'] is not None:
            parts.append(f"edx={r['str_edx']!r}")
        else:
            parts.append(f"edx=0x{r['edx']:x}")
        for a, s in zip(r['stack'], r['stack_strs']):
            if s is not None:
                parts.append(f"str={s!r}")
            else:
                parts.append(f"i32=0x{a:x}")
        lines.append("  ".join(parts))
    out = "\n".join(lines) + "\n"
    Path(out_path).write_text(out)
    print(f"wrote {out_path} ({len(lines)} calls)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dll")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or Path(args.dll).with_suffix(".script").name
    return run(args.dll, out)


if __name__ == "__main__":
    sys.exit(main())
