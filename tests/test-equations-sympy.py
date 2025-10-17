#!/usr/bin/env python3
import os
import sys
import importlib
from unittest import TestCase

# This test validates that we can extract equation blocks from the SymPy tests.
# It does not depend on sibling repos.

class TestExtractEquationsSympy(TestCase):
    def test_extract_equations_from_sympy_tests(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scripts_dir = os.path.join(root, 'scripts')
        sys.path.insert(0, scripts_dir)
        try:
            utils = importlib.import_module('_equation_test_utils')
        finally:
            sys.path.pop(0)

        blocks = utils.extract_all_test_sections(os.path.join(root, 'tests'))
        # Expect at least one block present in the SymPy tests
        self.assertIsInstance(blocks, list)
        self.assertGreaterEqual(len(blocks), 1, 'No equation blocks found in SymPy tests; expected at least one.')
