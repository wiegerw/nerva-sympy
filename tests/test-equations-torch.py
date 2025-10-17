#!/usr/bin/env python3
import os
import sys
import runpy
from unittest import TestCase

# This test runs the torch check-equations script. It requires the sibling repo nerva-torch
# to be present at ../nerva-torch with the expected layers.py path.


class TestCheckEquationsTorch(TestCase):
    def test_check_equations_torch(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sympy_scripts = os.path.join(root, 'scripts')
        script = os.path.join(sympy_scripts, 'check-equations-torch.py')

        # Locate sibling repo layers.py
        sibling_root = os.path.dirname(root)
        layers_py = os.path.join(sibling_root, 'nerva-torch', 'src', 'nerva_torch', 'layers.py')

        if not os.path.exists(layers_py):
            self.skipTest("Expected sibling repo not found at '../nerva-torch'. Please ensure the repo layout matches the monorepo expectations.")

        # Load script and call main()
        sys.path.insert(0, sympy_scripts)
        try:
            ns = runpy.run_path(script)
        finally:
            sys.path.pop(0)
        main = ns.get('main')
        self.assertIsNotNone(main, 'Could not find main() in check-equations-torch.py')

        # Verbosity via env var NERVA_CHECK_VERBOSE: '', 'v', 'vv'
        v = os.environ.get('NERVA_CHECK_VERBOSE', '')
        argv = ['-' + v] if v in ('v', 'vv') else []
        ret = int(main(argv))
        self.assertEqual(ret, 0)
