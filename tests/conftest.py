# SPDX-License-Identifier: Apache-2.0
"""Task 3.0c: the two CMSIS-NN oracle test suites (test_export_cmsisnn.py,
test_infer_t1.py, test_infer_t2.py) must be re-runnable against the Stage-B
release models (once Phase 4 trains them) without editing source — pass
--tier1-model/--tier2-model. Defaults to the Stage-A fixture models.
"""
import os

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RELEASE_TIER1 = os.path.join(_PROJECT_ROOT, "outputs", "models", "tier1_release.tflite")
_RELEASE_TIER2 = os.path.join(_PROJECT_ROOT, "outputs", "models", "tier2_release.tflite")
_FIXTURE_TIER1 = os.path.join(_PROJECT_ROOT, "outputs", "models", "tier1_fixture.tflite")
_FIXTURE_TIER2 = os.path.join(_PROJECT_ROOT, "outputs", "models", "tier2_fixture.tflite")

DEFAULT_TIER1_MODEL = _RELEASE_TIER1 if os.path.isfile(_RELEASE_TIER1) else _FIXTURE_TIER1
DEFAULT_TIER2_MODEL = _RELEASE_TIER2 if os.path.isfile(_RELEASE_TIER2) else _FIXTURE_TIER2


def pytest_addoption(parser):
    parser.addoption(
        "--tier1-model", action="store", default=None,
        help=f"Path to the Tier-1 .tflite model (default: {DEFAULT_TIER1_MODEL})",
    )
    parser.addoption(
        "--tier2-model", action="store", default=None,
        help=f"Path to the Tier-2 .tflite model (default: {DEFAULT_TIER2_MODEL})",
    )


@pytest.fixture(scope="session")
def tier1_model_path(request):
    return request.config.getoption("--tier1-model") or DEFAULT_TIER1_MODEL


@pytest.fixture(scope="session")
def tier2_model_path(request):
    return request.config.getoption("--tier2-model") or DEFAULT_TIER2_MODEL
