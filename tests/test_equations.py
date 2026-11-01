#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Check that the layer equations of the backends match the equations validated by the SymPy tests.

The check-equations scripts in the scripts directory read the source code of layers.py and compare
the feedforward and backpropagation equations textually with the equation blocks in the SymPy tests.
They do not import the backend or its framework, so these tests only need the source of each backend.

The source of a backend is located in the following order:

1. The environment variable NERVA_<BACKEND>_LAYERS_FILE, e.g. NERVA_TORCH_LAYERS_FILE=/path/to/layers.py.
2. A sibling checkout, e.g. ../nerva-torch/src/nerva_torch/layers.py.
3. An installed package, e.g. nerva_torch. The package is located without importing it, so it may
   have been installed without its framework (pip install --no-deps nerva-torch).

A backend whose source cannot be found is skipped.
"""

import importlib
import importlib.util
import os
import runpy
import sys
from typing import Optional
from unittest import TestCase

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, 'scripts')


def find_layers_file(backend: str) -> Optional[str]:
    """Return the path of layers.py of the given backend, or None if it cannot be found."""
    path = os.environ.get(f'NERVA_{backend.upper()}_LAYERS_FILE')
    if path:
        return path

    path = os.path.join(os.path.dirname(ROOT), f'nerva-{backend}', 'src', f'nerva_{backend}', 'layers.py')
    if os.path.exists(path):
        return path

    # For a top level package, find_spec does not execute the package.
    spec = importlib.util.find_spec(f'nerva_{backend}')
    if spec is not None and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            path = os.path.join(location, 'layers.py')
            if os.path.exists(path):
                return path

    return None


def run_check_equations_script(backend: str, extra_args=()) -> int:
    """Run scripts/check-equations-<backend>.py and return its exit code."""
    script = os.path.join(SCRIPTS_DIR, f'check-equations-{backend}.py')
    sys.path.insert(0, SCRIPTS_DIR)  # the scripts import _equation_test_utils
    try:
        namespace = runpy.run_path(script)
    finally:
        sys.path.pop(0)
    main = namespace.get('main')
    if main is None:
        raise RuntimeError(f'Could not find main() in {script}')

    # Verbosity via the environment variable NERVA_CHECK_VERBOSE: '', 'v', 'vv'
    verbosity = os.environ.get('NERVA_CHECK_VERBOSE', '')
    args = ['-' + verbosity] if verbosity in ('v', 'vv') else []
    return int(main(args + list(extra_args)))


class TestEquations(TestCase):
    def _check_backend(self, backend: str):
        layers_file = find_layers_file(backend)
        if layers_file is None:
            self.skipTest(f"Source of nerva-{backend} not found. Set NERVA_{backend.upper()}_LAYERS_FILE, "
                          f"use a sibling checkout ../nerva-{backend}, or install nerva-{backend}.")
        self.assertEqual(run_check_equations_script(backend, ['--layers-file', layers_file]), 0)

    def test_equations_numpy(self):
        self._check_backend('numpy')

    def test_equations_jax(self):
        self._check_backend('jax')

    def test_equations_tensorflow(self):
        self._check_backend('tensorflow')

    def test_equations_torch(self):
        self._check_backend('torch')

    def test_equations_sympy(self):
        # Checks the layers of nerva-sympy itself, so no other repositories are needed.
        self.assertEqual(run_check_equations_script('sympy'), 0)

    def test_extract_equations_from_sympy_tests(self):
        sys.path.insert(0, SCRIPTS_DIR)
        try:
            utils = importlib.import_module('_equation_test_utils')
        finally:
            sys.path.pop(0)

        blocks = utils.extract_all_test_sections(os.path.join(ROOT, 'tests'))
        self.assertIsInstance(blocks, list)
        self.assertGreaterEqual(len(blocks), 1, 'No equation blocks found in the SymPy tests; expected at least one.')


if __name__ == '__main__':
    import unittest
    unittest.main()
