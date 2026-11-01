#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Generate the files that are shared by the Python backends from the templates in nerva-sympy.

- templates/backend_tests/*.py are rendered to nerva-<backend>/tests/
- templates/backend_src/*.py are rendered to nerva-<backend>/src/nerva_<backend>/

The templates use the token nerva_BACKEND (and nerva-BACKEND), which is replaced by the name of the backend,
e.g. nerva_torch. Backend-specific behaviour lives in the tests/utilities.py file of each backend, which is not
generated. It provides the same functions for every backend, such as to_tensor, to_numpy, from_numpy,
set_parameter and make_optimizer.

Usage:
    python scripts/sync_backend_tests.py            # write the files to the ../nerva-<backend> checkouts
    python scripts/sync_backend_tests.py --check    # report differences, exit with status 1 if there are any
"""

import argparse
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT / 'templates' / 'backend_tests'
SRC_TEMPLATE_DIR = ROOT / 'templates' / 'backend_src'
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
    src_templates = sorted(SRC_TEMPLATE_DIR.glob('*.py'))
    if not templates:
        print(f'Error: no templates found in {TEMPLATE_DIR}')
        return 1

    differences = 0
    for backend in BACKENDS:
        tests_dir = args.root / f'nerva-{backend}' / 'tests'
        src_dir = args.root / f'nerva-{backend}' / 'src' / f'nerva_{backend}'
        for directory in (tests_dir, src_dir):
            if not directory.is_dir():
                print(f'Error: {directory} not found')
                return 1
        for template, target_dir in [(t, tests_dir) for t in templates] + [(t, src_dir) for t in src_templates]:
            expected = render(template.read_text(encoding='utf-8'), backend)
            target = target_dir / template.name
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
        print(f'All {len(templates) + len(src_templates)} shared files are in sync for {len(BACKENDS)} backends.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
