"""Builds the engine with `zig cc` (pip package `ziglang`).

zig cc is a full clang toolchain with bundled libc headers, so the same
command line cross-compiles for every supported board from any host.

    python tools/build.py                    # host build + run tests
    python tools/build.py --target aarch64   # Pi 3/4/5, Orange Pi 5, RK35xx (64-bit OS)
    python tools/build.py --target all       # every target, compile only
    python tools/build.py --debug            # -O0 -g with UBSan (zig default)

Outputs go to build/<target>/. On boards, the CMake build is the usual path;
this script is for development hosts and cross-checking all targets.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "engine"
PYPKG = ROOT / "newch"

# name -> (zig target triple, extra flags). Comments list typical boards.
TARGETS: dict[str, tuple[str | None, list[str]]] = {
    "host": (None, []),
    # Pi 3/4/5, Zero 2W, Orange Pi 5, Rock 5, Jetson (64-bit Linux)
    "aarch64": ("aarch64-linux-gnu", ["-mcpu=generic"]),
    # A76/A55 boards: enables dotprod + fp16 code paths at compile time
    "aarch64-v82": ("aarch64-linux-gnu", ["-mcpu=cortex_a55"]),
    # Pi 2/3/Zero 2W on 32-bit Raspberry Pi OS, older Allwinner boards
    "armv7": ("arm-linux-gnueabihf", ["-mcpu=cortex_a7"]),
    # VisionFive 2, Milk-V, BPI-F3 (RVV kernels: planned)
    "riscv64": ("riscv64-linux-gnu", []),
    "x86_64-linux": ("x86_64-linux-gnu", []),
    # Fully static (musl) builds: run under qemu-user in CI or copy to any board
    "aarch64-static": ("aarch64-linux-musl", ["-mcpu=generic"]),
    "armv7-static": ("arm-linux-musleabihf", ["-mcpu=cortex_a7"]),
    "riscv64-static": ("riscv64-linux-musl", []),
}

PROGRAMS = {
    "nc_tests": ENGINE / "tests" / "test_main.c",
    "train_spiral": ROOT / "examples" / "train_spiral.c",
}


def engine_sources() -> list[Path]:
    return sorted((ENGINE / "src").rglob("*.c"))


def build(target: str, debug: bool, lib_only: bool = False) -> Path:
    triple, extra = TARGETS[target]
    out_dir = ROOT / "build" / target
    out_dir.mkdir(parents=True, exist_ok=True)
    is_windows = triple is None and sys.platform == "win32"
    exe = ".exe" if is_windows else ""

    cflags = ["-std=c11", "-Wall", "-Wextra", "-Wno-unused-parameter", f"-I{ENGINE / 'include'}"]
    cflags += ["-O0", "-g"] if debug else ["-O2", "-fno-sanitize=undefined"]
    if triple:
        cflags += ["-target", triple]
    cflags += extra
    libs = [] if is_windows else ["-lm", "-lpthread"]

    zig = [sys.executable, "-m", "ziglang", "cc"]
    for name, main in PROGRAMS.items():
        if lib_only or not main.exists():
            continue
        out = out_dir / f"{name}{exe}"
        print(f"[{target}] {name}")
        subprocess.run([*zig, *cflags, *map(str, engine_sources()), str(main), "-o", str(out), *libs],
                       check=True, cwd=ROOT)

    # Shared library for the Python package (loaded with ctypes).
    # Only NC_API functions are exported; internals (nc__*) stay hidden.
    lib = out_dir / shared_lib_name(is_windows)
    shared = ["-DNC_BUILD_SHARED", "-fvisibility=hidden", *([] if is_windows else ["-fPIC"])]
    shared += [] if debug else ["-s"]  # strip: smaller, and only the public API is visible
    print(f"[{target}] {lib.name}")
    subprocess.run([*zig, *cflags, *shared, "-shared", *map(str, engine_sources()), "-o", str(lib), *libs],
                   check=True, cwd=ROOT)
    if triple is None:
        shutil.copy2(lib, PYPKG / lib.name)
    return out_dir


def shared_lib_name(windows: bool) -> str:
    if windows:
        return "nc.dll"
    return "libnc.dylib" if sys.platform == "darwin" else "libnc.so"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="host", choices=[*TARGETS, "all"])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-run", action="store_true", help="do not run tests after a host build")
    ap.add_argument("--lib-only", action="store_true", help="only build the shared library (used by pip)")
    args = ap.parse_args()

    targets = list(TARGETS) if args.target == "all" else [args.target]
    try:
        for t in targets:
            out_dir = build(t, args.debug, args.lib_only)
            if t == "host" and not (args.no_run or args.lib_only):
                exe = ".exe" if sys.platform == "win32" else ""
                subprocess.run([str(out_dir / f"nc_tests{exe}")], check=True)
                print("\n[python] unittest")
                subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                               check=True, cwd=ROOT)
    except subprocess.CalledProcessError as e:
        return e.returncode or 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
