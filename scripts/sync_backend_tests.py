#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Generate the test files that are shared by the Python backends from the templates in templates/backend_tests.

The templates use the token nerva_BACKEND (and nerva-BACKEND), which is replaced by the name of the backend,
e.g. nerva_torch. Backend-specific behaviour lives in the tests/utilities.py file of each backend, which is not
generated. It provides the same functions for every backend, such as to_tensor, to_numpy, from_numpy,
set_parameter and make_optimizer.

Usage:
    python scripts/sync_backend_tests.py            # write the test files to ../nerva-<backend>/tests
    python scripts/sync_backend_tests.py --check    # report differences, exit with status 1 if there are any
"""

import argparse
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT / 'templates' / 'backend_tests'
BACKENDS = ['numpy', 'jax', 'tensorflow', 'torch']


def render(template: str, backend: str) -> str:
    return template.replace('nerva_BACKEND', f'nerva_{backend}').replace('nerva-BACKEND', f'nerva-{backend}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='Only report differences; exit with status 1 if there are any')
    parser.add_argument('--root', type=Path, default=ROOT.parent,
                        help='Directory that contains the nerva-<backend> checkouts (default: the parent directory of nerva-sympy)')
    args = parser.parse_args()

    templates = sorted(TEMPLATE_DIR.glob('*.py'))
    if not templates:
        print(f'Error: no templates found in {TEMPLATE_DIR}')
        return 1

    differences = 0
    for backend in BACKENDS:
        tests_dir = args.root / f'nerva-{backend}' / 'tests'
        if not tests_dir.is_dir():
            print(f'Error: {tests_dir} not found')
            return 1
        for template in templates:
            expected = render(template.read_text(encoding='utf-8'), backend)
            target = tests_dir / template.name
            actual = target.read_text(encoding='utf-8') if target.exists() else ''
            if actual == expected:
                continue
            differences += 1
            if args.check:
                sys.stdout.writelines(difflib.unified_diff(actual.splitlines(keepends=True), expected.splitlines(keepends=True),
                                                           fromfile=str(target), tofile=f'{template.name} (from template)'))
            else:
                target.write_text(expected, encoding='utf-8')
                print(f'Updated {target}')

    if args.check:
        if differences:
            print(f'{differences} file(s) differ from the templates. Run scripts/sync_backend_tests.py to update them.')
            return 1
        print(f'All {len(templates)} shared test files are in sync for {len(BACKENDS)} backends.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
