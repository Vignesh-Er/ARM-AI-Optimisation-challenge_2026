# SPDX-License-Identifier: Apache-2.0
"""GATE 3.5 — Corstone-300 FVP Instruction Profiling Harness.

Performs two-point differencing instruction counts on the Cortex-M55/Helium
Corstone-300 Fast Model FVP.

This script automates:
1. Cross-compilation of the PACI benchmark binary using arm-none-eabi-gcc
   for Cortex-M55 + Helium (-mcpu=cortex-m55).
2. Execution under the Corstone-300 FVP (FVP_Corstone_SSE-300_Ethos-U55).
3. Extraction of instruction counts between timing markers to compute exact
   per-tier instruction costs.
4. Emission of Schema v2 JSON output with metric = "instructions".

Requirements:
- GNU Arm Embedded Toolchain (arm-none-eabi-gcc 13.x+)
- Arm Corstone-300 FVP (FVP_Corstone_SSE-300_Ethos-U55)
"""

import argparse
import json
import os
import re
import subprocess
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_toolchain():
    """Verify arm-none-eabi-gcc and FVP executable availability."""
    gcc_path = os.environ.get("ARM_GCC", "arm-none-eabi-gcc")
    fvp_path = os.environ.get("CORSTONE_FVP", "FVP_Corstone_SSE-300_Ethos-U55")

    gcc_ok = False
    try:
        res = subprocess.run([gcc_path, "--version"], capture_output=True, text=True)
        if res.returncode == 0:
            gcc_ok = True
    except FileNotFoundError:
        pass

    fvp_ok = False
    try:
        res = subprocess.run([fvp_path, "--version"], capture_output=True, text=True)
        if res.returncode == 0:
            fvp_ok = True
    except FileNotFoundError:
        pass

    return gcc_path, fvp_path, gcc_ok, fvp_ok


def parse_fvp_trace(trace_log):
    """Parse instruction trace log from Corstone-300 FVP output."""
    instructions = {}
    current_unit = None
    count_start = 0

    pattern = re.compile(r"\[BENCH_MARK\]\s+(\w+)\s+(START|END)\s+inst=(\d+)")
    with open(trace_log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                unit, phase, inst_str = m.group(1), m.group(2), int(m.group(3))
                if phase == "START":
                    current_unit = unit
                    count_start = inst_str
                elif phase == "END" and current_unit == unit:
                    delta = inst_str - count_start
                    instructions[unit] = delta
                    current_unit = None

    return instructions


def main():
    parser = argparse.ArgumentParser(description="Corstone-300 FVP Instruction Profiler")
    parser.add_argument("--output", "-o", default="outputs/bench/cortex-m55-fvp.json",
                        help="Output Schema v2 JSON path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check prerequisites and print launch commands without running")
    args = parser.parse_args()

    gcc_path, fvp_path, gcc_ok, fvp_ok = check_toolchain()

    print("=================================================================")
    print(" PACI GATE 3.5 — Corstone-300 FVP Instruction Profiler")
    print("=================================================================")
    print(f" arm-none-eabi-gcc: {'FOUND (' + gcc_path + ')' if gcc_ok else 'NOT FOUND'}")
    print(f" Corstone-300 FVP:  {'FOUND (' + fvp_path + ')' if fvp_ok else 'NOT FOUND'}")
    print("-----------------------------------------------------------------")

    if not gcc_ok or not fvp_ok:
        print("\n[NOTICE] FVP toolchain components are missing in this environment.")
        print("To run Cortex-M55 instruction profiling locally:")
        print(" 1. Install GNU Arm Embedded Toolchain: arm-none-eabi-gcc")
        print(" 2. Install Arm Ecosystem FVP: FVP_Corstone_SSE-300_Ethos-U55")
        print(" 3. See detailed instructions in docs/FVP_INSTRUCTIONS.md\n")
        if not args.dry_run:
            sys.exit(2)

    if args.dry_run:
        print("[DRY-RUN] Prerequisite checks complete.")
        sys.exit(0)

    # Cross-compile benchmark binary for Cortex-M55
    build_dir = os.path.join(_PROJECT_ROOT, "build_fvp")
    cmake_cmd = [
        "cmake", "-S", _PROJECT_ROOT, "-B", build_dir,
        f"-DCMAKE_C_COMPILER={gcc_path}",
        "-DCMAKE_SYSTEM_NAME=Generic",
        "-DCMAKE_SYSTEM_PROCESSOR=arm",
        "-DCMAKE_C_FLAGS=-mcpu=cortex-m55 -mfloat-abi=hard -mfpu=fpv5-d16 -O2 -fno-fast-math -ffp-contract=off",
        "-DPACI_BUILD_BENCH=ON"
    ]
    print("Building for Cortex-M55...")
    subprocess.run(cmake_cmd, check=True)
    subprocess.run(["cmake", "--build", build_dir, "--target", "bench_main"], check=True)

    elf_path = os.path.join(build_dir, "bench", "bench_main.elf")
    log_path = os.path.join(build_dir, "fvp_trace.log")

    fvp_cmd = [
        fvp_path,
        "-a", elf_path,
        "-C", "core_clk.params=0",
        "--stat",
        "-o", log_path
    ]
    print("Running Corstone-300 FVP simulation...")
    subprocess.run(fvp_cmd, check=True)

    instructions = parse_fvp_trace(log_path)
    print("Instruction count differencing results:")
    for unit, count in instructions.items():
        print(f"  {unit}: {count:,} instructions")

    # Format Schema v2 JSON
    out_dict = {
        "schema_version": 2,
        "target": "cortex-m55-fvp",
        "cpu": "Cortex-M55 + Helium",
        "compiler": "arm-none-eabi-gcc 13.2.1",
        "flags": "-mcpu=cortex-m55 -mfloat-abi=hard -O2 -fno-fast-math -ffp-contract=off",
        "metric": "instructions",
        "note": "Instruction counts via Corstone-300 FVP two-point differencing.",
        "batches": 1,
        "model_artifacts": {"tier1": "tier1_model.tflite", "tier2": "tier2_model.tflite"},
        "units": {
            unit: {"hot": {"value": count, "mad": 0.0}}
            for unit, count in instructions.items()
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)

    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
