# SPDX-License-Identifier: Apache-2.0
"""G2.5 gate: a fixture number (trained on the easy pre-Phase-4 dataset,
Task 2.0's throwaway Stage A models) must never reach a results artifact.
Fixture files are named *_fixture.* on purpose (see phase4_tinyml/train.py);
this just greps outputs/reports/, outputs/bench/, and README.md for the
substring "fixture" and fails if found.

Usage: python tools/check_no_fixture_in_results.py
Exit code 0: clean. Exit code 1: a results artifact mentions "fixture".
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SCAN_TARGETS = [
    os.path.join(_PROJECT_ROOT, "outputs", "reports"),
    os.path.join(_PROJECT_ROOT, "outputs", "bench"),
    os.path.join(_PROJECT_ROOT, "README.md"),
]


def _iter_files(target):
    if os.path.isfile(target):
        yield target
    elif os.path.isdir(target):
        for root, _dirs, files in os.walk(target):
            for name in files:
                yield os.path.join(root, name)


def find_fixture_leaks():
    leaks = []
    for target in _SCAN_TARGETS:
        for path in _iter_files(target):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for lineno, line in enumerate(f, start=1):
                        clean_line = line.lower().replace("check_no_fixture", "")
                        if "fixture" in clean_line:
                            leaks.append((path, lineno, line.strip()))
            except (UnicodeDecodeError, OSError):
                continue
    return leaks


def main():
    leaks = find_fixture_leaks()
    if leaks:
        print("FAIL: fixture references found in results artifacts:")
        for path, lineno, line in leaks:
            print(f"  {path}:{lineno}: {line}")
        sys.exit(1)
    print("OK: no fixture references in outputs/reports, outputs/bench, or README.md")
    sys.exit(0)


if __name__ == "__main__":
    main()
