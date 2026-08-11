# SPDX-License-Identifier: Apache-2.0
"""Task 3.0c: the two CMSIS-NN oracle test suites (test_export_cmsisnn.py,
test_infer_t1.py, test_infer_t2.py) must be re-runnable against the Stage-B
release models (once Phase 4 trains them) without editing source — pass
--tier1-model/--tier2-model. Defaults to the Stage-A fixture models.
"""
import os

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TIER1_MODEL = os.path.join(_PROJECT_ROOT, "outputs", "models", "tier1_fixture.tflite")
DEFAULT_TIER2_MODEL = os.path.join(_PROJECT_ROOT, "outputs", "models", "tier2_fixture.tflite")


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
