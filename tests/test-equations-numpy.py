#!/usr/bin/env python3
import os
import sys
import runpy
from unittest import TestCase

# This test runs the numpy check-equations script. It requires the sibling repo nerva-numpy
# to be present at ../nerva-numpy with the expected layers.py path.


class TestCheckEquationsNumpy(TestCase):
    def test_check_equations_numpy(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(root, 'scripts', 'check-equations-numpy.py')

        sibling_root = os.path.dirname(root)
        layers_py = os.path.join(sibling_root, 'nerva-numpy', 'src', 'nerva_numpy', 'layers.py')

        if not os.path.exists(layers_py):
            self.skipTest("Expected sibling repo not found at '../nerva-numpy'. Please ensure the repo layout matches the monorepo expectations.")

        sys.path.insert(0, os.path.join(root, 'scripts'))
        try:
            ns = runpy.run_path(script)
        finally:
            sys.path.pop(0)
        main = ns.get('main')
        self.assertIsNotNone(main, 'Could not find main() in check-equations-numpy.py')

        v = os.environ.get('NERVA_CHECK_VERBOSE', '')
        argv = ['-' + v] if v in ('v', 'vv') else []
        ret = int(main(argv))
        self.assertEqual(ret, 0)
