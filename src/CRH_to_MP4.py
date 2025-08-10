import os, sys, struct, wave, shutil, subprocess
# ---------- Proven defaults ----------
A_BYTES         = 2942          # per-frame audio bytes
OFFSET_BITS     = 16
DEINT           = "even-odd"    # even-odd | odd-even | halves
TILE_BITS_DEF   = "field-rev"   # preferred mode; decode uses this

# Audio decode tuning
FINE_STEP       = 16
COARSE_STEP     = 512
LSB_COARSE      = 0             # 0 => LSB=0→coarse(512), LSB=1→fine(16)
INCLUDE_SEED    = True          # seed sample(s) at frame start to avoid clicks

# ---------- Bit helpers ----------
class BRlsb:
    __slots__=("b","p","bp","n")
    def __init__(self,data): self.b=memoryview(data); self.n=len(data); self.p=0; self.bp=0
    def _ok(self):
        if self.p>=self.n: raise EOFError("BitReader: out of data")
    def bit(self):
        self._ok(); v=(self.b[self.p]>>self.bp)&1; self.bp+=1
        if self.bp==8: self.bp=0; self.p+=1
        return v
    def bits(self,n):
        v=0
        for i in range(n): v|=(self.bit()<<i)
        return v
    def bits_safe(self,n, default=0):
        try: return self.bits(n)
        except EOFError: return default

class BRmsb:
    __slots__=("b","p","bp","n")
    def __init__(self,data): self.b=memoryview(data); self.n=len(data); self.p=0; self.bp=0
    def _ok(self):
        if self.p>=self.n: raise EOFError("BitReaderMSB: out of data")
    def bit(self):
        self._ok(); v=(self.b[self.p]>>(7-self.bp))&1; self.bp+=1
        if self.bp==8: self.bp=0; self.p+=1
        return v
    def bits(self,n):
        v=0
        for _ in range(n): v=(v<<1)|self.bit()
        return v
    def bits_safe(self,n, default=0):
        try: return self.bits(n)
        except EOFError: return default

_REV = bytes(int(f"{i:08b}"[::-1],2) for i in range(256))
def byterev(b:bytes)->bytes: return bytes(_REV[x] for x in b)
def rbits(v,n):
    r=0
    for i in range(n): r=(r<<1)|((v>>i)&1)
    return r
def five8(v): return (v<<3)|(v>>2)
def clip5(x): return 0 if x<0 else (31 if x>31 else x)

# ---------- Header ----------
def parse_header(f):
    f.seek(0)
    ver=int.from_bytes(f.read(2),"little")
    ah=f.read(48)
    if len(ah)<48: raise ValueError("Bad CRH header")
    sr=int.from_bytes(ah[0:4],"little")
    ch=int.from_bytes(ah[14:16],"little")
    frames=int.from_bytes(f.read(2),"little")
    fps=int.from_bytes(f.read(2),"little") or 15
    return {"ver":ver,"sr":sr,"ch":ch,"frames":frames,"fps":fps,"pos":f.tell()}

# ---------- Component decode (previous FRAME prediction) ----------
def dec_comp_framepred(data, plane, offbits, prev_frame_plane):
    br=BRlsb(data); dst=bytearray(plane); pos=0
    prev = prev_frame_plane if prev_frame_plane is not None else bytes(plane)
    while pos<plane:
        try: flag=br.bit()
        except EOFError: break
        if flag==0:
            val=br.bits_safe(5,0); ln=br.bits_safe(8,0)
            n=min(ln, plane-pos)
            if n>0: dst[pos:pos+n]=bytes([val])*n; pos+=n
        else:
            sid=br.bits_safe(1,0); off=br.bits_safe(offbits,0); ln=br.bits_safe(8,0)
            src = dst if sid==0 else prev
            for i in range(ln):
                if pos>=plane: break
                b=src[off+i] if 0<=off+i<len(src) else 0
                dst[pos]=b; pos+=1
    return dst

# ---------- Sampling / color ----------
def sample_even_odd(plane,x,y,w): idx=2*((y//2)*w+x)+(y&1); return plane[idx]&31
def sample_odd_even(plane,x,y,w): idx=2*((y//2)*w+x)+((y^1)&1); return plane[idx]&31
def sample_halves(plane,x,y,w): half=len(plane)//2; idx=(y%2)*half+(y//2)*w+x; return plane[idx]&31

def reconstruct_rgb(c0,c1,c2,tdata,w,h,tilebits,deint):
    if tilebits=="byte-rev":
        br=BRmsb(byterev(tdata)); tw=br.bits_safe(8,20); th=br.bits_safe(8,20)
        def fields(): return br.bits_safe(2,3), br.bits_safe(4,0)*4, br.bits_safe(4,0)*4, br.bits_safe(4,8)*2, br.bits_safe(4,8)*2
    else:
        br=BRlsb(tdata); tw=rbits(br.bits_safe(8,0),8); th=rbits(br.bits_safe(8,0),8)
        if tw==0: tw=20
        if th==0: th=20
        def fields(): return rbits(br.bits_safe(2,0),2), rbits(br.bits_safe(4,0),4)*4, rbits(br.bits_safe(4,0),4)*4, rbits(br.bits_safe(4,8),4)*2, rbits(br.bits_safe(4,8),4)*2
    tiles_x=max(1, w//max(1,tw)); tiles_y=max(1, h//max(1,th))
    sampler = sample_even_odd if deint=="even-odd" else (sample_odd_even if deint=="odd-even" else sample_halves)
    row=w*3; img=bytearray(h*row)
    def tab(step,off): return [int(i*step/32 + off - 16) for i in range(32)]
    first=fields(); it=0
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            if it==0: m,s1,s2,o1,o2=first
            else:     m,s1,s2,o1,o2=fields()
            T1,T2=tab(s1,o1),tab(s2,o2)
            x0=tx*tw; y0=ty*th
            for yy in range(th):
                if y0+yy>=h: break
                dstrow=(y0+yy)*row
                for xx in range(tw):
                    if x0+xx>=w: break
                    x=x0+xx; y=y0+yy
                    a=sampler(c0,x,y,w); b=sampler(c1,x,y,w); c=sampler(c2,x,y,w)
                    if   m==0: r=clip5(a+T2[c]); g=clip5(b+T1[c]); b5=c
                    elif m==1: r=clip5(b+T1[c]); g=c;           b5=clip5(a+T2[c])
                    elif m==2: r=c;           g=clip5(a+T2[c]); b5=clip5(b+T1[c])
                    else:      r=a;           g=b;             b5=c
                    p=dstrow+x*3; img[p:p+3]=bytes([five8(r),five8(g),five8(b5)])
            it+=1
    return bytes(img)

# ---------- Audio (no gain anywhere) ----------
def sign7(v):  # 0..127 -> -64..63
    return v-128 if (v & 0x40) else v

def decode_audio_block(raw: bytes, ch: int, a_bytes: int) -> bytes:
    mv=memoryview(raw); p=0
    prev=[0,0]
    if ch==1:
        if len(mv) < 2: return b""
        prev[0]=struct.unpack_from("<h", mv, p)[0]; p+=2
    else:
        if len(mv) < 4: return b""
        prev[0]=struct.unpack_from("<h", mv, p)[0]; p+=2
        prev[1]=struct.unpack_from("<h", mv, p)[0]; p+=2

    deltas = max(0, a_bytes - 2*ch)
    samples = deltas // max(1,ch)

    out = bytearray()
    if INCLUDE_SEED:
        if ch==1:
            s = max(-32768, min(32767, prev[0]))
            out += struct.pack("<h", s)
        else:
            L = max(-32768, min(32767, prev[0]))
            R = max(-32768, min(32767, prev[1]))
            out += struct.pack("<hh", L, R)

    for _ in range(samples):
        for c in range(ch):
            if p>=len(mv): break
            b = mv[p]; p+=1
            step = (COARSE_STEP if (b & 1)==0 else FINE_STEP) if LSB_COARSE==0 else ((COARSE_STEP if (b & 1) else FINE_STEP))
            dv = sign7(b >> 1) * step
            s = prev[c] + dv
            s = max(-32768, min(32767, s))
            prev[c] = s
            out += struct.pack("<h", s)
    return bytes(out)

# ---------- ffmpeg (with auto-install for imageio-ffmpeg) ----------
def ensure_ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe: return exe
    try:
        import imageio_ffmpeg as iioff
    except Exception:
        print("[info] Installing imageio-ffmpeg (one-time)...")
        cmd = [sys.executable, "-m", "pip", "install", "--user", "imageio-ffmpeg"]
        try:
            subprocess.check_call(cmd)
        except Exception as e:
            print("[error] pip install failed:", e)
    try:
        import imageio_ffmpeg as iioff
        return iioff.get_ffmpeg_exe()
    except Exception:
        return None

# ---------- Dimension detection ----------
def detect_dims(in_path, hdr_pos):
    with open(in_path,"rb") as f:
        f.seek(hdr_pos)
        if len(f.read(A_BYTES))!=A_BYTES: return 320,200,TILE_BITS_DEF
        h16=f.read(16)
        if len(h16)<16: return 320,200,TILE_BITS_DEF
        _full,c0,c1,c2=struct.unpack("<IIII",h16)
        comp0=f.tell(); tpos=comp0+c0+c1+c2
        f.seek(tpos); raw=f.read(4)
        if len(raw)<4: return 320,200,TILE_BITS_DEF
        tsize=int.from_bytes(raw,"little")
        if tsize<=0 or tsize>50_000_000: return 320,200,TILE_BITS_DEF
        tdata=f.read(tsize)
        if len(tdata)!=tsize: return 320,200,TILE_BITS_DEF

    def dims_from_tdata(mode):
        if mode=="field-rev":
            br=BRlsb(tdata); tw=rbits(br.bits_safe(8,0),8); th=rbits(br.bits_safe(8,0),8)
        else:
            br=BRmsb(byterev(tdata)); tw=br.bits_safe(8,0); th=br.bits_safe(8,0)
        if not tw or not th: return None
        rem_bits=len(tdata)*8 - 16
        if rem_bits<=0 or rem_bits%18!=0: return None
        nt=rem_bits//18
        best=None
        for x in range(1, int(nt**0.5)+1):
            if nt % x: continue
            for tx,ty in ((x, nt//x), (nt//x, x)):
                w,h = tw*tx, th*ty
                if 32<=w<=4096 and 32<=h<=2160:
                    ar_diff = abs((w/h) - (4/3))
                    score=(ar_diff, -w*h)
                    if best is None or score < best[0]:
                        best=(score,(w,h))
        return (*best[1], mode) if best else None

    dims = dims_from_tdata(TILE_BITS_DEF) or dims_from_tdata("byte-rev")
    if not dims: return 320,200,TILE_BITS_DEF
    return dims  # (w,h,tilebits)

# ---------- Convert one file ----------
def convert(in_path):
    base = os.path.splitext(os.path.basename(in_path))[0]
    folder = os.path.dirname(in_path)
    out_mp4 = os.path.join(folder, base + ".mp4")
    wav_path = os.path.join(folder, base + ".wav")

    with open(in_path,"rb") as f: hdr=parse_header(f)
    sr = hdr["sr"]; ch = max(1, hdr["ch"])
    fps = hdr["fps"] or 15
    width,height,tilebits = detect_dims(in_path, hdr["pos"])

    # Pass A: WAV (audio first)
    with wave.open(wav_path,"wb") as wf:
        wf.setnchannels(ch); wf.setsampwidth(2); wf.setframerate(sr)
        with open(in_path,"rb") as f:
            f.seek(hdr["pos"])
            while True:
                aud=f.read(A_BYTES)
                if len(aud)!=A_BYTES: break
                wf.writeframes(decode_audio_block(aud, ch, A_BYTES))
                # skip video block
                h16=f.read(16)
                if len(h16)<16: break
                _full,c0,c1,c2=struct.unpack("<IIII",h16)
                f.seek(c0+c1+c2,1)
                b=f.read(4)
                if len(b)<4: break
                tsize=int.from_bytes(b,"little")
                f.seek(tsize,1)

    # Pass B: send raw RGB to ffmpeg and mux audio
    ff = ensure_ffmpeg_exe()
    if not ff:
        raise RuntimeError("ffmpeg not found. Install ffmpeg or allow imageio-ffmpeg to install.")

    cmd=[ff,"-y",
         "-f","rawvideo","-pix_fmt","rgb24","-s",f"{width}x{height}","-r",str(fps),"-i","-",
         "-i", wav_path,
         "-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac","-b:a","128k","-shortest", out_mp4]
    proc=subprocess.Popen(cmd, stdin=subprocess.PIPE)

    plane=width*height
    prev0=prev1=prev2=None
    with open(in_path,"rb") as f:
        f.seek(hdr["pos"])
        while True:
            aud=f.read(A_BYTES)
            if len(aud)!=A_BYTES: break
            h16=f.read(16)
            if len(h16)<16: break
            _full,c0,c1,c2=struct.unpack("<IIII",h16)
            comp0=f.tell(); comp1=comp0+c0; comp2=comp1+c1; tpos=comp2+c2
            f.seek(comp0); c0b=f.read(c0)
            f.seek(comp1); c1b=f.read(c1)
            f.seek(comp2); c2b=f.read(c2)
            f.seek(tpos); tsize=int.from_bytes(f.read(4),"little"); tdata=f.read(tsize)

            c0r=dec_comp_framepred(c0b,plane,OFFSET_BITS,prev0)
            c1r=dec_comp_framepred(c1b,plane,OFFSET_BITS,prev1)
            c2r=dec_comp_framepred(c2b,plane,OFFSET_BITS,prev2)
            prev0,prev1,prev2 = c0r,c1r,c2r

            rgb = reconstruct_rgb(c0r,c1r,c2r,tdata, width,height, TILE_BITS_DEF, DEINT)
            try:
                proc.stdin.write(rgb)
            except Exception:
                break

    proc.stdin.close(); proc.wait()
    try: os.remove(wav_path)
    except Exception: pass
    print(f"[info] {base}: dims={width}x{height} fps={fps} tilebits={tilebits}")
    return out_mp4

# ---------- Entry ----------
if __name__=="__main__":
    if len(sys.argv)<2:
        print("Drag & drop one or more .CRH files onto this script.")
        sys.exit(0)

    ff = ensure_ffmpeg_exe()
    if not ff:
        print("[fatal] Couldn’t find or install ffmpeg. Install ffmpeg or rerun to allow imageio-ffmpeg install.")
        sys.exit(1)

    for path in sys.argv[1:]:
        try:
            out = convert(path)
            print(f"[done] {out}")
        except Exception as e:
            print(f"[error] {path}: {e}")
