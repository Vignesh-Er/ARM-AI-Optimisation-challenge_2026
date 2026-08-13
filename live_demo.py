#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
PACI Live Demonstration Entrypoint — Arm AI Optimization Challenge 2026
=======================================================================
Launcher script executing the primary C-core driven live demonstration (demo_live.py).
"""

import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import demo_live

if __name__ == "__main__":
    sys.exit(demo_live.run_c_cascade_demo())
