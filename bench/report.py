# SPDX-License-Identifier: Apache-2.0
"""Benchmark Reporting Tool for PACI-Arm (Schema v2).

Reads a benchmark JSON file conforming to Schema v2 (bench/bench_json.h/c),
validates all fields strictly, and produces:
  1. A structured Markdown benchmark report (Execution Latency, Cascade Trace,
     Memory Footprint, and Platform Metadata).
  2. Latency and cascade breakdown plots (if matplotlib is available).

Usage:
  python bench/report.py
  python bench/report.py outputs/bench/aarch64-linux.json
  python bench/report.py --input outputs/bench/aarch64-linux.json --output-md outputs/reports/bench_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


def _validate_number(val: Any, name: str, min_val: Optional[float] = 0.0) -> float:
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        raise ValueError(f"Expected numeric value for '{name}', got {type(val).__name__} ({val!r})")
    num = float(val)
    if min_val is not None and num < min_val:
        raise ValueError(f"Value for '{name}' must be >= {min_val}, got {num}")
    return num


def _validate_int(val: Any, name: str, min_val: Optional[int] = 0) -> int:
    if not isinstance(val, int) or isinstance(val, bool):
        raise ValueError(f"Expected integer value for '{name}', got {type(val).__name__} ({val!r})")
    if min_val is not None and val < min_val:
        raise ValueError(f"Value for '{name}' must be >= {min_val}, got {val}")
    return int(val)


def _validate_str(val: Any, name: str, allow_empty: bool = False) -> str:
    if not isinstance(val, str):
        raise ValueError(f"Expected string for '{name}', got {type(val).__name__} ({val!r})")
    if not allow_empty and not val.strip():
        raise ValueError(f"String '{name}' cannot be empty")
    return val


def _validate_dict(val: Any, name: str) -> Dict[str, Any]:
    if not isinstance(val, dict):
        raise ValueError(f"Expected object/dict for '{name}', got {type(val).__name__} ({val!r})")
    return val


def validate_and_parse_schema_v2(data: Any) -> Dict[str, Any]:
    """Strictly validates benchmark JSON against Schema v2.

    Raises ValueError or KeyError if any required field is missing or malformed.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Benchmark JSON root must be an object, got {type(data).__name__}")

    # Top-level required fields per Schema v2 specification
    required_top_keys = [
        "schema_version",
        "target",
        "cpu",
        "compiler",
        "flags",
        "metric",
        "note",
        "batches",
        "model_artifacts",
        "units",
    ]
    for k in required_top_keys:
        if k not in data:
            raise KeyError(f"Missing required top-level field in Schema v2 benchmark JSON: '{k}'")

    schema_version = _validate_int(data["schema_version"], "schema_version", min_val=1)
    if schema_version != 2:
        raise ValueError(f"Unsupported schema_version {schema_version}; expected 2")

    target = _validate_str(data["target"], "target")
    cpu = _validate_str(data["cpu"], "cpu", allow_empty=True)
    compiler = _validate_str(data["compiler"], "compiler", allow_empty=True)
    flags = _validate_str(data["flags"], "flags", allow_empty=True)
    metric = _validate_str(data["metric"], "metric", allow_empty=True)
    note = _validate_str(data["note"], "note", allow_empty=True)
    batches = _validate_int(data["batches"], "batches", min_val=1)

    model_artifacts = _validate_dict(data["model_artifacts"], "model_artifacts")
    for tier in ("tier1", "tier2"):
        if tier not in model_artifacts:
            raise KeyError(f"Missing '{tier}' in model_artifacts")
        _validate_str(model_artifacts[tier], f"model_artifacts.{tier}")

    units = _validate_dict(data["units"], "units")
    required_units = ["tier0_ekf", "tier1_int4", "tier2_int8", "cascade_trace"]
    for u in required_units:
        if u not in units:
            raise KeyError(f"Missing required unit in units object: '{u}'")

    parsed_units: Dict[str, Any] = {}

    # Validate unit measurements: tier0_ekf, tier1_int4, tier2_int8
    for unit_name in ("tier0_ekf", "tier1_int4", "tier2_int8"):
        u_dict = _validate_dict(units[unit_name], f"units.{unit_name}")
        if "hot" not in u_dict:
            raise KeyError(f"Missing 'hot' measurement in units.{unit_name}")
        hot_dict = _validate_dict(u_dict["hot"], f"units.{unit_name}.hot")
        if "value" not in hot_dict or "mad" not in hot_dict:
            raise KeyError(f"units.{unit_name}.hot must contain 'value' and 'mad'")

        hot_val = _validate_number(hot_dict["value"], f"units.{unit_name}.hot.value")
        hot_mad = _validate_number(hot_dict["mad"], f"units.{unit_name}.hot.mad")

        cold_data: Optional[Dict[str, float]] = None
        if "cold" in u_dict and u_dict["cold"] is not None:
            cold_dict = _validate_dict(u_dict["cold"], f"units.{unit_name}.cold")
            if "value" not in cold_dict or "mad" not in cold_dict:
                raise KeyError(f"units.{unit_name}.cold must contain 'value' and 'mad'")
            cold_val = _validate_number(cold_dict["value"], f"units.{unit_name}.cold.value")
            cold_mad = _validate_number(cold_dict["mad"], f"units.{unit_name}.cold.mad")
            cold_data = {"value": cold_val, "mad": cold_mad}

        scratch_bytes: Optional[int] = None
        if "scratch_bytes" in u_dict and u_dict["scratch_bytes"] is not None:
            scratch_bytes = _validate_int(u_dict["scratch_bytes"], f"units.{unit_name}.scratch_bytes", min_val=-1)

        parsed_units[unit_name] = {
            "hot": {"value": hot_val, "mad": hot_mad},
            "cold": cold_data,
            "scratch_bytes": scratch_bytes,
        }

    # Validate cascade_trace
    casc_dict = _validate_dict(units["cascade_trace"], "units.cascade_trace")
    for k in ("total", "n1", "n2", "steps"):
        if k not in casc_dict:
            raise KeyError(f"Missing '{k}' in units.cascade_trace")

    casc_total = _validate_number(casc_dict["total"], "units.cascade_trace.total")
    casc_n1 = _validate_int(casc_dict["n1"], "units.cascade_trace.n1")
    casc_n2 = _validate_int(casc_dict["n2"], "units.cascade_trace.n2")
    casc_steps = _validate_int(casc_dict["steps"], "units.cascade_trace.steps", min_val=1)

    parsed_units["cascade_trace"] = {
        "total": casc_total,
        "n1": casc_n1,
        "n2": casc_n2,
        "steps": casc_steps,
    }

    # Parse optional footprint
    parsed_footprint: Optional[Dict[str, Any]] = None
    if "footprint" in data and data["footprint"] is not None:
        fp_dict = _validate_dict(data["footprint"], "footprint")
        parsed_variants: Dict[str, Dict[str, int]] = {}
        if "variants" in fp_dict and fp_dict["variants"] is not None:
            v_dict = _validate_dict(fp_dict["variants"], "footprint.variants")
            for v_name, v_data in v_dict.items():
                v_data_dict = _validate_dict(v_data, f"footprint.variants.{v_name}")
                text = _validate_int(v_data_dict.get("text", v_data_dict.get("text_bytes", 0)), f"footprint.variants.{v_name}.text")
                data_sec = _validate_int(v_data_dict.get("data", v_data_dict.get("data_bytes", 0)), f"footprint.variants.{v_name}.data")
                bss = _validate_int(v_data_dict.get("bss", v_data_dict.get("bss_bytes", 0)), f"footprint.variants.{v_name}.bss")
                total = v_data_dict.get("total", v_data_dict.get("total_bytes", text + data_sec + bss))
                total = _validate_int(total, f"footprint.variants.{v_name}.total")
                parsed_variants[v_name] = {
                    "text": text,
                    "data": data_sec,
                    "bss": bss,
                    "total": total,
                }

        parsed_deltas: Dict[str, Dict[str, int]] = {}
        if "per_tier_delta" in fp_dict and fp_dict["per_tier_delta"] is not None:
            d_dict = _validate_dict(fp_dict["per_tier_delta"], "footprint.per_tier_delta")
            for d_name, d_data in d_dict.items():
                d_data_dict = _validate_dict(d_data, f"footprint.per_tier_delta.{d_name}")
                text = _validate_int(d_data_dict.get("text", d_data_dict.get("text_bytes", 0)), f"footprint.per_tier_delta.{d_name}.text")
                data_sec = _validate_int(d_data_dict.get("data", d_data_dict.get("data_bytes", 0)), f"footprint.per_tier_delta.{d_name}.data")
                bss = _validate_int(d_data_dict.get("bss", d_data_dict.get("bss_bytes", 0)), f"footprint.per_tier_delta.{d_name}.bss")
                total = d_data_dict.get("total", d_data_dict.get("total_bytes", text + data_sec + bss))
                total = _validate_int(total, f"footprint.per_tier_delta.{d_name}.total")
                parsed_deltas[d_name] = {
                    "text": text,
                    "data": data_sec,
                    "bss": bss,
                    "total": total,
                }
        elif parsed_variants and "core_only" in parsed_variants:
            core = parsed_variants["core_only"]
            if "core+tier1" in parsed_variants:
                t1 = parsed_variants["core+tier1"]
                parsed_deltas["tier1"] = {
                    "text": max(0, t1["text"] - core["text"]),
                    "data": max(0, t1["data"] - core["data"]),
                    "bss": max(0, t1["bss"] - core["bss"]),
                    "total": max(0, t1["total"] - core["total"]),
                }
            if "core+tier2" in parsed_variants:
                t2 = parsed_variants["core+tier2"]
                parsed_deltas["tier2"] = {
                    "text": max(0, t2["text"] - core["text"]),
                    "data": max(0, t2["data"] - core["data"]),
                    "bss": max(0, t2["bss"] - core["bss"]),
                    "total": max(0, t2["total"] - core["total"]),
                }

        parsed_footprint = {
            "variants": parsed_variants,
            "per_tier_delta": parsed_deltas,
        }

    return {
        "schema_version": schema_version,
        "target": target,
        "cpu": cpu,
        "compiler": compiler,
        "flags": flags,
        "metric": metric,
        "note": note,
        "batches": batches,
        "model_artifacts": model_artifacts,
        "units": parsed_units,
        "footprint": parsed_footprint,
    }


def _format_bytes(n_bytes: Optional[int]) -> str:
    if n_bytes is None or n_bytes < 0:
        return "N/A"
    if n_bytes == 0:
        return "0 B"
    if n_bytes >= 1024 * 1024:
        return f"{n_bytes:,} B ({n_bytes / (1024 * 1024):.2f} MB)"
    if n_bytes >= 1024:
        return f"{n_bytes:,} B ({n_bytes / 1024:.2f} KB)"
    return f"{n_bytes:,} B"


def _format_time_ns(ns: float, metric: str) -> str:
    unit = "ns" if "ns" in metric.lower() else ("cycles" if "cycle" in metric.lower() else metric)
    if unit == "ns":
        if ns >= 1_000_000:
            return f"{ns / 1_000_000:.3f} ms ({ns:,.1f} ns)"
        if ns >= 1_000:
            return f"{ns / 1_000:.2f} µs ({ns:,.1f} ns)"
        return f"{ns:.2f} ns"
    return f"{ns:,.2f} {unit}"


def generate_markdown_report(data: Dict[str, Any]) -> str:
    """Generates a complete GitHub-flavored Markdown benchmark report."""
    target = data["target"]
    cpu = data["cpu"]
    compiler = data["compiler"]
    flags = data["flags"]
    metric = data["metric"]
    note = data["note"]
    batches = data["batches"]
    artifacts = data["model_artifacts"]
    units = data["units"]
    footprint = data["footprint"]

    unit_label = "ns" if "ns" in metric.lower() else ("cycles" if "cycle" in metric.lower() else metric)

    # Latencies
    t0_hot = units["tier0_ekf"]["hot"]["value"]
    t0_mad = units["tier0_ekf"]["hot"]["mad"]

    t1_hot = units["tier1_int4"]["hot"]["value"]
    t1_hot_mad = units["tier1_int4"]["hot"]["mad"]
    t1_cold_obj = units["tier1_int4"]["cold"]
    t1_cold_str = f"{t1_cold_obj['value']:.2f} ± {t1_cold_obj['mad']:.2f} {unit_label}" if t1_cold_obj else "—"
    t1_scratch = units["tier1_int4"]["scratch_bytes"]

    t2_hot = units["tier2_int8"]["hot"]["value"]
    t2_hot_mad = units["tier2_int8"]["hot"]["mad"]
    t2_cold_obj = units["tier2_int8"]["cold"]
    t2_cold_str = f"{t2_cold_obj['value']:.2f} ± {t2_cold_obj['mad']:.2f} {unit_label}" if t2_cold_obj else "—"
    t2_scratch = units["tier2_int8"]["scratch_bytes"]

    # Cascade trace calculations
    casc = units["cascade_trace"]
    steps = casc["steps"]
    total_time = casc["total"]
    n1 = casc["n1"]
    n2 = casc["n2"]

    n1_rate_pct = (n1 / steps) * 100.0
    n2_rate_pct = (n2 / steps) * 100.0
    t0_filter_pct = 100.0 - n1_rate_pct
    t1_filter_pct = 100.0 - (n2 / n1 * 100.0) if n1 > 0 else 100.0

    effective_step_cost = total_time / steps
    theoretical_step_cost = t0_hot + (n1 / steps) * t1_hot + (n2 / steps) * t2_hot
    always_on_tier2_cost = t0_hot + t2_hot

    savings_pct = (
        ((always_on_tier2_cost - effective_step_cost) / always_on_tier2_cost * 100.0)
        if always_on_tier2_cost > 0
        else 0.0
    )
    speedup = (always_on_tier2_cost / effective_step_cost) if effective_step_cost > 0 else 1.0

    lines: List[str] = [
        f"# PACI Benchmark Report: `{target}`",
        "",
        f"> Automated hardware benchmark evaluation conforming to **Schema v2** on **{target}**.",
        "",
        "## 1. Execution Latency",
        "",
        f"Measured across **{batches} batches** using median and Median Absolute Deviation (MAD).",
        "",
        f"| Tier / Execution Unit | Algorithm / Hardware Kernel | Hot Latency (Median ± MAD) | Cold Latency (Median ± MAD) | Scratch Buffer |",
        f"|:---|:---|:---:|:---:|:---:|",
        f"| **Tier 0 (EKF)** | Physics-residual scalar EKF + statistical gating | `{t0_hot:.2f} ± {t0_mad:.2f} {unit_label}` | — | N/A (0 B) |",
        f"| **Tier 1 (INT4)** | CMSIS-NN 4-bit Conv1D classifier (`arm_convolve_s4`) | `{t1_hot:.2f} ± {t1_hot_mad:.2f} {unit_label}` | `{t1_cold_str}` | {_format_bytes(t1_scratch)} |",
        f"| **Tier 2 (INT8)** | CMSIS-NN 8-bit Conv1D classifier (`arm_convolve_wrapper_s8`) | `{t2_hot:.2f} ± {t2_hot_mad:.2f} {unit_label}` | `{t2_cold_str}` | {_format_bytes(t2_scratch)} |",
        "",
        "## 2. Cascade Trace Results",
        "",
        f"End-to-end evaluation across standard **{steps:,} steps** input trace (`k=1.0` benchmark sequence).",
        "",
        "| Cascade Metric | Realized Value | Performance Analysis |",
        "|:---|:---:|:---|",
        f"| **Total Evaluation Time** | `{_format_time_ns(total_time, metric)}` | Cumulative trace execution time |",
        f"| **Total Steps Evaluated** | `{steps:,}` | Standard deterministic benchmark length |",
        f"| **Tier 1 Invocations ($N_1$)** | `{n1:,}` | Escalation rate: **{n1_rate_pct:.2f}%** ({t0_filter_pct:.2f}% filtered by Tier 0) |",
        f"| **Tier 2 Invocations ($N_2$)** | `{n2:,}` | Escalation rate: **{n2_rate_pct:.2f}%** ({t1_filter_pct:.2f}% resolved by Tier 1) |",
        f"| **Effective Per-Step Latency** | `{effective_step_cost:.2f} {unit_label}/step` | Realized amortized execution cost per sample |",
        f"| **Theoretical Decomposed Cost** | `{theoretical_step_cost:.2f} {unit_label}/step` | $T_0 + \\frac{{N_1}}{{N}} T_1 + \\frac{{N_2}}{{N}} T_2$ model prediction |",
        f"| **Always-On Tier 2 Baseline** | `{always_on_tier2_cost:.2f} {unit_label}/step` | Un-gated Tier 2 execution on every step |",
        f"| **Compute Latency Reduction** | **{savings_pct:.2f}%** | Relative savings vs Always-On Tier 2 baseline |",
        f"| **Effective Speedup Factor** | **{speedup:.2f}×** | Realized acceleration from adaptive tri-tier gating |",
        "",
        "## 3. Memory & Static Footprint",
        "",
    ]

    if footprint and footprint.get("variants"):
        variants = footprint["variants"]
        deltas = footprint.get("per_tier_delta", {})
        lines.extend([
            "### Binary Configuration Variants",
            "",
            "| Variant Configuration | Text (Flash) | Data (Init) | BSS (Zero-Init) | Total Static Footprint | Scope / Description |",
            "|:---|:---:|:---:|:---:|:---:|:---|",
        ])
        for v_name, v_data in variants.items():
            desc = {
                "core_only": "Physics model, scalar EKF, ring buffer only",
                "core+tier1": "Core + INT4 CMSIS-NN inference engine",
                "core+tier2": "Core + INT8 CMSIS-NN inference engine",
                "full": "Complete tri-tier cascade pipeline",
            }.get(v_name, "Target configuration build")
            lines.append(
                f"| `{v_name}` | `{_format_bytes(v_data['text'])}` | `{_format_bytes(v_data['data'])}` | `{_format_bytes(v_data['bss'])}` | `{_format_bytes(v_data['total'])}` | {desc} |"
            )
        lines.append("")

        if deltas:
            lines.extend([
                "### Incremental Per-Tier Memory Delta",
                "",
                "| Subsystem Tier | Incremental Text (Flash) | Incremental Data | Incremental BSS (RAM) | Total Incremental Cost |",
                "|:---|:---:|:---:|:---:|:---:|",
            ])
            for d_name, d_data in deltas.items():
                label = "Tier 1 (INT4)" if "1" in d_name else ("Tier 2 (INT8)" if "2" in d_name else d_name)
                lines.append(
                    f"| **{label}** | `+{_format_bytes(d_data['text'])}` | `+{_format_bytes(d_data['data'])}` | `+{_format_bytes(d_data['bss'])}` | `+{_format_bytes(d_data['total'])}` |"
                )
            lines.append("")

    lines.extend([
        "### Scratch Buffer Allocations",
        "",
        "| Tier Component | Scratch Memory Allocation | Storage Location | Reusability Semantics |",
        "|:---|:---:|:---|:---|",
        f"| **Tier 0 (EKF)** | `0 B` | Stack | Stack-allocated scalar operations only |",
        f"| **Tier 1 (INT4)** | `{_format_bytes(t1_scratch)}` | Static RAM Arena | Reusable activation arena (shared sequentially) |",
        f"| **Tier 2 (INT8)** | `{_format_bytes(t2_scratch)}` | Static RAM Arena | Reusable activation arena (shared sequentially) |",
        "",
    ])

    if not (footprint and footprint.get("variants")):
        lines.extend([
            "> [!NOTE]",
            "> Static binary section sizes (`.text`, `.data`, `.bss`) were not recorded in this benchmark artifact. Scratch buffer requirements are reported directly from execution unit metadata.",
            "",
        ])

    lines.extend([
        "## 4. Platform & Build Metadata",
        "",
        "| Configuration Property | Value | Description / Scope |",
        "|:---|:---|:---|",
        f"| **Target Platform** | `{target}` | Hardware benchmark deployment target |",
        f"| **CPU / Core** | `{cpu}` | Host or target microarchitecture |",
        f"| **Compiler** | `{compiler}` | Toolchain used for C harness compilation |",
        f"| **Compiler Flags** | `{flags}` | Optimization and IEEE-754 compliance flags |",
        f"| **Primary Metric** | `{metric}` | Timing metric reported by benchmark units |",
        f"| **Measurement Batches** | `{batches}` | Statistical sample count for median / MAD |",
        f"| **Tier 1 Artifact** | `{artifacts['tier1']}` | INT4 model flatbuffer / weight source |",
        f"| **Tier 2 Artifact** | `{artifacts['tier2']}` | INT8 model flatbuffer / weight source |",
        f"| **Schema Version** | `v{data['schema_version']}` | PACI Benchmark JSON Schema specification |",
        f"| **Harness Notes** | {note} | Operational notes and execution details |",
        "",
    ])

    return "\n".join(lines)


def generate_plots(data: Dict[str, Any], output_path: str) -> bool:
    """Generates execution latency and cascade breakdown visual plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is not installed; skipping plot generation.")
        return False

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    units = data["units"]
    metric = data["metric"]
    unit_label = "ns" if "ns" in metric.lower() else ("cycles" if "cycle" in metric.lower() else metric)

    t0_hot = units["tier0_ekf"]["hot"]["value"]
    t0_mad = units["tier0_ekf"]["hot"]["mad"]

    t1_hot = units["tier1_int4"]["hot"]["value"]
    t1_hot_mad = units["tier1_int4"]["hot"]["mad"]
    t1_cold_obj = units["tier1_int4"]["cold"]
    t1_cold = t1_cold_obj["value"] if t1_cold_obj else 0.0
    t1_cold_mad = t1_cold_obj["mad"] if t1_cold_obj else 0.0

    t2_hot = units["tier2_int8"]["hot"]["value"]
    t2_hot_mad = units["tier2_int8"]["hot"]["mad"]
    t2_cold_obj = units["tier2_int8"]["cold"]
    t2_cold = t2_cold_obj["value"] if t2_cold_obj else 0.0
    t2_cold_mad = t2_cold_obj["mad"] if t2_cold_obj else 0.0

    casc = units["cascade_trace"]
    steps = casc["steps"]
    total_time = casc["total"]
    n1 = casc["n1"]
    n2 = casc["n2"]

    effective_cost = total_time / steps
    always_on_cost = t0_hot + t2_hot

    t0_contrib = t0_hot
    t1_contrib = (n1 / steps) * t1_hot
    t2_contrib = (n2 / steps) * t2_hot

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=200)

    # Subplot 1: Per-Unit Latency Comparison
    labels = ["Tier 0 (EKF)\nHot", "Tier 1 (INT4)\nHot", "Tier 1 (INT4)\nCold", "Tier 2 (INT8)\nHot", "Tier 2 (INT8)\nCold"]
    values = [t0_hot, t1_hot, t1_cold, t2_hot, t2_cold]
    errors = [t0_mad, t1_hot_mad, t1_cold_mad, t2_hot_mad, t2_cold_mad]
    colors = ["#0284C7", "#0D9488", "#14B8A6", "#E11D48", "#F43F5E"]

    # Filter out empty cold values if not present
    valid_indices = [i for i, v in enumerate(values) if v > 0]
    filt_labels = [labels[i] for i in valid_indices]
    filt_values = [values[i] for i in valid_indices]
    filt_errors = [errors[i] for i in valid_indices]
    filt_colors = [colors[i] for i in valid_indices]

    bars = ax1.bar(
        filt_labels,
        filt_values,
        yerr=filt_errors,
        capsize=5,
        color=filt_colors,
        edgecolor="#1E293B",
        linewidth=1.2,
        alpha=0.9,
    )

    for bar in bars:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.1f} {unit_label}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#0F172A",
        )

    ax1.set_title("Per-Tier Execution Latency (Median ± MAD)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel(f"Latency ({unit_label})", fontsize=10, fontweight="semibold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)

    # Subplot 2: Stacked Cascade Cost vs Always-On Baseline
    cascade_labels = ["Always-On Tier 2", "PACI Cascade\n(Amortized Realized)"]
    x_pos = [0, 1]

    # Always-On bar
    ax2.bar(
        [0],
        [always_on_cost],
        width=0.45,
        label="Always-On Tier 2",
        color="#E11D48",
        edgecolor="#1E293B",
        linewidth=1.2,
        alpha=0.9,
    )
    ax2.annotate(
        f"{always_on_cost:.1f} {unit_label}\n(100%)",
        xy=(0, always_on_cost),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

    # Stacked PACI bar
    b0 = ax2.bar([1], [t0_contrib], width=0.45, label="Tier 0 (EKF, 100% steps)", color="#0284C7", edgecolor="#1E293B", linewidth=1.2)
    b1 = ax2.bar([1], [t1_contrib], bottom=[t0_contrib], width=0.45, label=f"Tier 1 (INT4, {n1/steps*100:.1f}% steps)", color="#0D9488", edgecolor="#1E293B", linewidth=1.2)
    b2 = ax2.bar([1], [t2_contrib], bottom=[t0_contrib + t1_contrib], width=0.45, label=f"Tier 2 (INT8, {n2/steps*100:.1f}% steps)", color="#F59E0B", edgecolor="#1E293B", linewidth=1.2)

    total_stack = t0_contrib + t1_contrib + t2_contrib
    reduction_pct = (1.0 - effective_cost / always_on_cost) * 100.0 if always_on_cost > 0 else 0.0
    speedup = (always_on_cost / effective_cost) if effective_cost > 0 else 1.0

    ax2.annotate(
        f"{effective_cost:.1f} {unit_label}\n(-{reduction_pct:.1f}%, {speedup:.1f}×)",
        xy=(1, total_stack),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#0F172A",
    )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(cascade_labels, fontsize=10, fontweight="semibold")
    ax2.set_title(f"Effective Per-Step Cost vs Baseline ({reduction_pct:.1f}% Compute Reduction)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel(f"Cost per Step ({unit_label})", fontsize=10, fontweight="semibold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.suptitle(f"PACI Benchmark Performance Summary ({data['target']})", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Generated benchmark plot: {output_path}")
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="PACI-Arm Schema v2 Benchmark Report Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_pos",
        nargs="?",
        default=None,
        help="Positional path to input Schema v2 benchmark JSON.",
    )
    parser.add_argument(
        "-i", "--input",
        dest="input_flag",
        default=None,
        help="Path to input Schema v2 benchmark JSON.",
    )
    parser.add_argument(
        "-o", "--output-md",
        default="outputs/reports/bench_report.md",
        help="Path to write the output Markdown report.",
    )
    parser.add_argument(
        "-p", "--output-plot",
        default="outputs/plots/bench_breakdown.png",
        help="Path to write the output breakdown plot (PNG).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable graphical plot generation.",
    )

    args = parser.parse_args(argv)

    input_path = args.input_flag or args.input_pos or "outputs/bench/aarch64-linux.json"
    output_md_path = args.output_md
    output_plot_path = args.output_plot

    if not os.path.isfile(input_path):
        print(f"[ERROR] Input benchmark file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as err:
        print(f"[ERROR] Malformed JSON in '{input_path}': {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"[ERROR] Failed to read '{input_path}': {err}", file=sys.stderr)
        return 1

    try:
        validated_data = validate_and_parse_schema_v2(raw_data)
    except (ValueError, KeyError) as err:
        print(f"[ERROR] Benchmark schema validation failed for '{input_path}': {err}", file=sys.stderr)
        return 1

    # Generate and write Markdown report
    try:
        md_content = generate_markdown_report(validated_data)
        os.makedirs(os.path.dirname(os.path.abspath(output_md_path)), exist_ok=True)
        with open(output_md_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(md_content + "\n")
        print(f"[INFO] Generated Markdown report: {output_md_path}")
    except Exception as err:
        print(f"[ERROR] Failed to write Markdown report to '{output_md_path}': {err}", file=sys.stderr)
        return 1

    # Generate plot unless requested otherwise
    if not args.no_plot and output_plot_path:
        generate_plots(validated_data, output_plot_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
