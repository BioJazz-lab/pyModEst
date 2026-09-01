"""Run every example and report whether each one still works.

The documentation quotes output from these scripts, so this is what keeps the
docs honest. Run it after changing the library:

    uv run python docs/examples/run_all.py
    uv run python docs/examples/run_all.py --show   # print each script's output
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = sorted(p for p in HERE.glob("[0-9][0-9]_*.py"))


def main() -> int:
    show = "--show" in sys.argv
    failures = 0
    print(f"running {len(SCRIPTS)} example(s) with {sys.executable}\n")
    for script in SCRIPTS:
        started = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, script.name], cwd=HERE,
            capture_output=True, text=True,
        )
        elapsed = time.perf_counter() - started
        ok = proc.returncode == 0
        failures += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {script.name:<28} {elapsed:6.1f}s")
        if show and ok:
            print("\n" + proc.stdout.rstrip() + "\n")
        if not ok:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:], file=sys.stderr)

    print(f"\n{len(SCRIPTS) - failures}/{len(SCRIPTS)} example(s) passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
