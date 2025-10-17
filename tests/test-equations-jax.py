#!/usr/bin/env python3
import os
import sys
import runpy
from unittest import TestCase

# This test runs the jax check-equations script. It requires the sibling repo nerva-jax
# to be present at ../nerva-jax with the expected layers.py path.


class TestCheckEquationsJax(TestCase):
    def test_check_equations_jax(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(root, 'scripts', 'check-equations-jax.py')

        sibling_root = os.path.dirname(root)
        layers_py = os.path.join(sibling_root, 'nerva-jax', 'src', 'nerva_jax', 'layers.py')

        if not os.path.exists(layers_py):
            self.skipTest("Expected sibling repo not found at '../nerva-jax'. Please ensure the repo layout matches the monorepo expectations.")

        sys.path.insert(0, os.path.join(root, 'scripts'))
        try:
            ns = runpy.run_path(script)
        finally:
            sys.path.pop(0)
        main = ns.get('main')
        self.assertIsNotNone(main, 'Could not find main() in check-equations-jax.py')

        v = os.environ.get('NERVA_CHECK_VERBOSE', '')
        argv = ['-' + v] if v in ('v', 'vv') else []
        ret = int(main(argv))
        self.assertEqual(ret, 0)
