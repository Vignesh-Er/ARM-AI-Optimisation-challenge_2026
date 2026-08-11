# SPDX-License-Identifier: Apache-2.0
"""G3.0: outputs/models/requant_sites.json must exist, be well-formed, and
match what tools/write_requant_sites.py would regenerate — Phase 5 imports
this file rather than re-deriving accumulation depths/scales itself, so it
can't be allowed to drift from the exporter the way paci_params.h could
from config.py (D4) without a check like this one.
"""
import json
import os
import sys

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
from tools.write_requant_sites import build_requant_sites  # noqa: E402

_SITES_PATH = os.path.join(_PROJECT_ROOT, "outputs", "models", "requant_sites.json")
_REQUIRED_FIELDS = {"layer_name", "op", "accumulation_depth", "input_scale", "output_scale", "weight_bits"}


def _require_fixture_models():
    tier1 = os.path.join(_PROJECT_ROOT, "outputs", "models", "tier1_fixture.tflite")
    tier2 = os.path.join(_PROJECT_ROOT, "outputs", "models", "tier2_fixture.tflite")
    if not (os.path.isfile(tier1) and os.path.isfile(tier2)):
        pytest.skip("fixture models not found. Run `python tools/train_fixture_models.py` first.")


def test_requant_sites_json_exists_and_well_formed():
    if not os.path.isfile(_SITES_PATH):
        pytest.skip(f"{_SITES_PATH} not found. Run `python tools/write_requant_sites.py` first.")

    with open(_SITES_PATH) as f:
        data = json.load(f)

    assert "tier1" in data and "tier2" in data
    for tier in ("tier1", "tier2"):
        assert len(data[tier]) > 0
        for site in data[tier]:
            assert _REQUIRED_FIELDS.issubset(site.keys()), f"{tier} site missing fields: {site}"
            assert isinstance(site["accumulation_depth"], int) and site["accumulation_depth"] > 0
            if site["op"] != "MEAN":
                assert site["weight_bits"] in (4, 8)


def test_requant_sites_json_matches_current_export():
    _require_fixture_models()
    with open(_SITES_PATH) as f:
        committed = json.load(f)

    fresh = build_requant_sites()

    for tier in ("tier1", "tier2"):
        committed_names = [s["layer_name"] for s in committed[tier]]
        fresh_names = [s["layer_name"] for s in fresh[tier]]
        assert committed_names == fresh_names, (
            f"{tier}: requant_sites.json is stale relative to the current export. "
            "Run `python tools/write_requant_sites.py` and commit the result."
        )
        for committed_site, fresh_site in zip(committed[tier], fresh[tier]):
            assert committed_site["accumulation_depth"] == fresh_site["accumulation_depth"]
            assert committed_site["weight_bits"] == fresh_site["weight_bits"]
