# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Cross-framework tests: compare the four numerical backends with SymPy and with each other.

These tests are not run by default. They need all four backends (pip install "nerva-sympy[all]"), and are run with

    python tests/run_all_tests.py --run --cross-framework

or with pytest tests/cross_framework.
"""

import os
import sys

# Make the shared test utilities in the parent directory importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
