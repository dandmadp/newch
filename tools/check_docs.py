"""Runs the code examples of the user guide (guide/*.md) so they cannot go stale.

    python tools/check_docs.py [page.md ...]

Every ```python block runs; blocks of one page share a namespace and run in order, in a fresh
default context and a temporary working directory. ```py blocks are excerpts and are skipped.
Exit status 1 if any page fails.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NC_DEBUG_POISON", "1")  # use-after-free shows up as garbage

BLOCK = re.compile(r"^```python[^\n]*\n(.*?)^```", re.S | re.M)


def run_page(path: Path) -> bool:
    import newch

    blocks = BLOCK.findall(path.read_text(encoding="utf-8"))
    if not blocks:
        return True
    newch.set_default_context(newch.Context(seed=0))
    namespace: dict = {"__name__": "__docs__"}
    cwd = os.getcwd()
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            for i, code in enumerate(blocks, 1):
                try:
                    exec(compile(code, f"{path.name}[block {i}]", "exec"), namespace)
                except Exception:
                    print(f"FAIL {path.name}, block {i}:\n" + "".join("    " + line for line in code.splitlines(True)))
                    traceback.print_exc()
                    return False
        finally:
            os.chdir(cwd)
            newch.step_end()
    print(f"ok   {path.name} ({len(blocks)} blocks, {time.perf_counter() - t0:.1f}s)")
    return True


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pages = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "guide").glob("*.md"))
    failed = [p.name for p in pages if not run_page(p)]
    if failed:
        print("failed:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
