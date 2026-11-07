"""Run unit and integration checks in separate Python/JAX processes."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    for suite in ('unit', 'integration'):
        print(f'[test_suite] {suite}', flush=True)
        result = subprocess.run([sys.executable, '-m', 'pytest', f'tests/{suite}', '-q', *sys.argv[1:]], cwd=ROOT)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
