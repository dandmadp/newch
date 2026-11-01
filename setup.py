"""Builds the C engine into the package during `pip install` / `pip wheel`.

The library is loaded with ctypes (no Python C API), so wheels are tagged
py3-none-<platform>: one wheel per OS/CPU works for every Python 3 version.
"""

import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

try:
    from wheel.bdist_wheel import bdist_wheel
except ImportError:  # building an sdist only
    bdist_wheel = None

ROOT = Path(__file__).resolve().parent
LIB_NAMES = ("nc.dll", "libnc.so", "libnc.dylib")


class BuildWithEngine(build_py):
    def run(self):
        subprocess.run([sys.executable, str(ROOT / "tools" / "build.py"), "--lib-only"], check=True, cwd=ROOT)
        super().run()
        out = Path(self.build_lib) / "newch"
        for name in LIB_NAMES:
            src = ROOT / "newch" / name
            if src.exists():
                self.copy_file(str(src), str(out / name))


class BinaryDistribution(Distribution):
    def has_ext_modules(self):  # platform-specific wheel
        return True


cmdclass = {"build_py": BuildWithEngine}

if bdist_wheel is not None:
    class AnyPythonWheel(bdist_wheel):
        def get_tag(self):
            _, _, plat = super().get_tag()
            return "py3", "none", plat

    cmdclass["bdist_wheel"] = AnyPythonWheel

setup(cmdclass=cmdclass, distclass=BinaryDistribution)
