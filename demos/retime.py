#!/usr/bin/env python
# Slow down (or speed up) the demo animations without re-recording.
#
# Animated-WebP playback speed is just the per-frame delay stored in each ANMF
# chunk, so we can rescale it in place: same frames, same pixels, same file
# size, only the timing changes. VHS bakes in whatever the tapes dictate
# (TypingSpeed 40ms etc.), which plays fast; this multiplies every frame delay
# by FACTOR (>1 = slower).
#
# Always scales from the pristine capture in output/.orig/ so repeated runs
# don't compound. Usage:
#   uv run python demos/retime.py 1.4            # all tapes, 1.4x slower
#   uv run python demos/retime.py 1.4 hero usage # just these
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "output" / ".orig"
DST = HERE / "output"


def retime(data: bytes, factor: float) -> tuple[bytes, int, int]:
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP", "not a RIFF/WebP file"
    out = bytearray(data)
    off = 12
    before = after = 0
    while off + 8 <= len(out):
        cid = bytes(out[off : off + 4])
        size = struct.unpack("<I", out[off + 4 : off + 8])[0]
        body = off + 8
        if cid == b"ANMF":
            d = out[body + 12] | (out[body + 13] << 8) | (out[body + 14] << 16)
            nd = min(0xFFFFFF, max(1, round(d * factor)))
            before += d
            after += nd
            out[body + 12] = nd & 0xFF
            out[body + 13] = (nd >> 8) & 0xFF
            out[body + 14] = (nd >> 16) & 0xFF
        off = body + size + (size & 1)
    return bytes(out), before, after


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: retime.py FACTOR [names...]")
    factor = float(sys.argv[1])
    names = sys.argv[2:]
    files = (
        [SRC / f"{n}.webp" for n in names]
        if names
        else sorted(SRC.glob("*.webp"))
    )
    for src in files:
        if not src.exists():
            sys.exit(f"no pristine source at {src} (run record-all first)")
        out, before, after = retime(src.read_bytes(), factor)
        (DST / src.name).write_bytes(out)
        print(f"{src.stem:9} {before/1000:6.2f}s -> {after/1000:6.2f}s")


if __name__ == "__main__":
    main()
