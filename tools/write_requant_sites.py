# SPDX-License-Identifier: Apache-2.0
"""Task 3.0a: write outputs/models/requant_sites.json — an ordered list of
every requantization site in both models, with the accumulation depth,
input/output scale, and weight bit-width Phase 5's bias budget needs.
Phase 5 imports this file rather than re-deriving it from the exporter.

accumulation_depth is the number of int32 accumulation terms feeding one
output element before its single requantize call:
  conv:  HK * WK * C_IN
  pool:  PACI_WINDOW_SIZE (every window position contributes to the sum)
  FC:    C_IN

Usage: python tools/write_requant_sites.py
"""
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from tools.export_cmsisnn import export_tier1, export_tier2  # noqa: E402

_OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "outputs", "models", "requant_sites.json")
_TIER1_TFLITE = os.path.join(_PROJECT_ROOT, "outputs", "models", "tier1_fixture.tflite")
_TIER2_TFLITE = os.path.join(_PROJECT_ROOT, "outputs", "models", "tier2_fixture.tflite")

_SITE_FIELDS = ["name", "op", "accumulation_depth", "input_scale", "output_scale", "weight_bits"]


def _sites_for_model(layers):
    sites = []
    for layer in layers:
        site = {field: layer.get(field) for field in _SITE_FIELDS}
        site["layer_name"] = site.pop("name")
        sites.append(site)
    return sites


def build_requant_sites():
    if not os.path.isfile(_TIER1_TFLITE) or not os.path.isfile(_TIER2_TFLITE):
        raise FileNotFoundError(
            "Fixture models not found. Run `python tools/train_fixture_models.py` "
            "and `python tools/export_cmsisnn.py ... --prefix {tier1,tier2}` first."
        )

    tier1_layers = export_tier1(_TIER1_TFLITE, config.WINDOW_SIZE)
    tier2_layers = export_tier2(_TIER2_TFLITE, config.WINDOW_SIZE)

    return {
        "window_size": config.WINDOW_SIZE,
        "tier1": _sites_for_model(tier1_layers),
        "tier2": _sites_for_model(tier2_layers),
    }


def main():
    data = build_requant_sites()
    os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)
    with open(_OUTPUT_PATH, "w", newline="\n", encoding="ascii") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {_OUTPUT_PATH}")
    for tier in ("tier1", "tier2"):
        print(f"  {tier}:")
        for site in data[tier]:
            print(f"    {site['layer_name']:16s} {site['op']:20s} "
                  f"depth={site['accumulation_depth']} bits={site['weight_bits']}")


if __name__ == "__main__":
    main()
