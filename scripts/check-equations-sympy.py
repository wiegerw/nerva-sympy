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
    block_distance
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS_FILE = os.path.join(ROOT, 'src', 'nerva_sympy', 'layers.py')
TESTS_DIR = os.path.join(ROOT, 'tests')


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Check consistency of layer equations with tests')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity (use -vv for debug)')
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    layers = extract_layer_equations_generic(LAYERS_FILE)
    blocks = extract_all_test_sections()

    logging.info(f'Found {len(layers)} layers with equations to check.')
    logging.info(f'Found {len(blocks)} test equation blocks across files.')

    missing: List[str] = []
    for layer_name, eqs in layers.items():
        best_block = min(blocks, key=lambda b: block_distance(eqs, b)) if blocks else None

        def section_matches(section: str) -> bool:
            if not best_block:
                return False
            return sorted(eqs.get(section, [])) == sorted(best_block.get(section, []))

        ff_ok = section_matches('feedforward')
        bp_ok = section_matches('backpropagation')

        if ff_ok and bp_ok and best_block:
            meta = best_block.get('__meta__', {})
            logging.info(f'Layer {layer_name}: matched test block in {meta.get("file")} (block #{meta.get("index")})')
        else:
            logging.warning(f'Layer {layer_name}: no matching test block found')
            missing.append(layer_name)

    if missing:
        print('No matching test equations found for the following layers:')
        for name in missing:
            print(f' - {name}')
        print('\nDetails:')
        for name in missing:
            print(f'[{name}] feedforward:')
            for line in layers[name]['feedforward']:
                print('  ' + line)
            print(f'[{name}] backpropagation:')
                
            for line in layers[name]['backpropagation']:
                print('  ' + line)
        return 1
    else:
        print('Success: Matching test equations were found for all layers.')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
