# SPDX-License-Identifier: Apache-2.0
"""
Verifies that the benchmark section in README.md matches the latest generated
benchmark report. Can also update the README.md automatically.
"""

import argparse
import sys
import os
import re

def main():
    parser = argparse.ArgumentParser(description="Verify or update README.md benchmark section")
    parser.add_argument("--readme", default="README.md", help="Path to README.md")
    parser.add_argument("--report", default="outputs/reports/bench_report.md", help="Path to generated markdown report")
    parser.add_argument("--update", action="store_true", help="Update the README.md if out of date")
    args = parser.parse_args()

    if not os.path.exists(args.readme):
        print(f"[ERROR] README file not found: {args.readme}", file=sys.stderr)
        return 1

    if not os.path.exists(args.report):
        print(f"[ERROR] Benchmark report not found: {args.report}", file=sys.stderr)
        return 1

    with open(args.report, "r", encoding="utf-8") as f:
        report_content = f.read().strip()

    with open(args.readme, "r", encoding="utf-8") as f:
        readme_content = f.read()

    pattern = re.compile(r"(<!-- BENCHMARK:BEGIN -->\s*)(.*?)(\s*<!-- BENCHMARK:END -->)", re.DOTALL)
    match = pattern.search(readme_content)

    if not match:
        print("[ERROR] Could not find <!-- BENCHMARK:BEGIN --> and <!-- BENCHMARK:END --> tags in README.md", file=sys.stderr)
        return 1

    current_benchmark_section = match.group(2).strip()

    if current_benchmark_section == report_content:
        print("[INFO] README.md benchmark section is up to date.")
        return 0

    if args.update:
        new_readme = readme_content[:match.start(2)] + "\n" + report_content + "\n" + readme_content[match.end(2):]
        with open(args.readme, "w", encoding="utf-8") as f:
            f.write(new_readme)
        print("[INFO] README.md benchmark section updated successfully.")
        return 0
    else:
        print("[ERROR] README.md benchmark section is out of date!", file=sys.stderr)
        print(f"Run `python {sys.argv[0]} --update` to synchronize it.", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
