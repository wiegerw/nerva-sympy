#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import os
import argparse
import logging
from typing import Dict, List, Optional

from _equation_test_utils import (
    extract_all_test_sections,
    extract_layer_equations_generic,
    setup_logging,
    block_distance,
    print_block_equations,
    normalize_tf_transpose_calls,
)

SYMPY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIBLING_ROOT = os.path.dirname(SYMPY_ROOT)
LAYERS_FILE_DEFAULT = os.path.join(SIBLING_ROOT, 'nerva-tensorflow', 'src', 'nerva_tensorflow', 'layers.py')
TESTS_DIR_DEFAULT = os.path.join(SYMPY_ROOT, 'tests')


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Check consistency of nerva-tensorflow layer equations with SymPy tests')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity (use -vv for debug)')
    parser.add_argument('--layers-file', default=LAYERS_FILE_DEFAULT, help='Path to layers.py to inspect (default: ../nerva-tensorflow/src/nerva_tensorflow/layers.py)')
    parser.add_argument('--tests-dir', default=TESTS_DIR_DEFAULT, help='Path to SymPy tests directory (default: ./tests)')
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    layers = extract_layer_equations_generic(args.layers_file, replace_matmul_with_mul=True, extra_normalizers=[normalize_tf_transpose_calls])
    blocks = extract_all_test_sections(args.tests_dir)

    logging.info(f'Found {len(layers)} layers with equations to check.')
    logging.info(f'Found {len(blocks)} test equation blocks across files.')

    any_mismatch = False
    for layer_name, eqs in layers.items():
        best_block = min(blocks, key=lambda b: block_distance(eqs, b)) if blocks else None

        def section_matches(section: str) -> bool:
            if not best_block:
                return False
            return sorted(eqs.get(section, [])) == sorted(best_block.get(section, []))

        ff_ok = section_matches('feedforward')
        bp_ok = section_matches('backpropagation')

        if ff_ok and bp_ok:
            meta = best_block.get('__meta__', {}) if best_block else {}
            logging.info(f'Layer {layer_name}: equations match (from {meta.get("file")} block #{meta.get("index")})')
            if args.verbose:
                print(f'[{layer_name}] equations from layers.py:')
                print('  feedforward:')
                for line in eqs.get('feedforward', []):
                    print('    ' + line)
                print('  backpropagation:')
                for line in eqs.get('backpropagation', []):
                    print('    ' + line)
                if best_block:
                    print_block_equations(f'[{layer_name}] matching SymPy test', best_block)
            continue

        any_mismatch = True
        meta = best_block.get('__meta__', {}) if best_block else {}
        logging.warning(f'Layer {layer_name}: mismatch against SymPy (closest {meta.get("file")} block #{meta.get("index")})')

        def print_section(title: str, ours: List[str], theirs: List[str]):
            print(f'[{layer_name}] {title}:')
            print('  ours:')
            for line in ours:
                print('    ' + line)
            if best_block:
                print('  sympy:')
                for line in theirs:
                    print('    ' + line)

        if args.verbose:
            print_section('feedforward', eqs.get('feedforward', []), best_block.get('feedforward', []) if best_block else [])
            print_section('backpropagation', eqs.get('backpropagation', []), best_block.get('backpropagation', []) if best_block else [])
        else:
            if not ff_ok:
                print_section('feedforward', eqs.get('feedforward', []), best_block.get('feedforward', []) if best_block else [])
            if not bp_ok:
                print_section('backpropagation', eqs.get('backpropagation', []), best_block.get('backpropagation', []) if best_block else [])

    if any_mismatch:
        return 1
    else:
        print('Success: Matching test equations were found for all layers.')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
