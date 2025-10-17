#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import os
import re
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR_DEFAULT = os.path.join(ROOT, 'tests')

# Variables considered in the SymPy test blocks
EQ_VARS = {
    'feedforward': {'R', 'Sigma', 'inv_sqrt_Sigma', 'Z', 'Y'},
    'backpropagation': {'DZ', 'DW', 'Db', 'DX', 'Dbeta', 'Dgamma', 'Dal', 'Dar', 'Dtl', 'Dtr'},
}

_ws_re = re.compile(r"\s+")

def norm_expr(s: str) -> str:
    # Normalize expressions as they appear in SymPy tests
    s = s.split('#', 1)[0]
    s = s.strip()
    s = _ws_re.sub(' ', s)
    s = s.replace(' .T', '.T')
    # normalize parentheses around RHS to avoid trivial mismatches
    s = re.sub(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\((.*)\)$", r"\1 = \2", s)
    return s


def extract_test_sections_from_file(path: str) -> List[Dict[str, List[str]]]:
    """Parse a SymPy test file and extract equation blocks under '# feedforward' and '# backpropagation'.
    Returns a list of blocks, each block is a dict with keys 'feedforward' and 'backpropagation'
    mapping to lists of normalized assignment lines.
    """
    blocks: List[Dict[str, List[str]]] = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    block_idx = 0
    while i < n:
        if lines[i].strip().startswith('# feedforward'):
            block = {'feedforward': [], 'backpropagation': [], '__meta__': {'file': path, 'index': block_idx}}
            block_idx += 1
            i += 1
            # collect feedforward assignments until blank line or next comment starting with '#'
            while i < n and lines[i].strip() and not lines[i].lstrip().startswith('#'):
                line = lines[i]
                if '=' in line:
                    block['feedforward'].append(norm_expr(line))
                i += 1
            # advance to backpropagation
            while i < n and not lines[i].strip().startswith('# backpropagation'):
                i += 1
            if i < n and lines[i].strip().startswith('# backpropagation'):
                i += 1
                while i < n and lines[i].strip() and not lines[i].lstrip().startswith('#'):
                    line = lines[i]
                    if '=' in line:
                        block['backpropagation'].append(norm_expr(line))
                    i += 1
            blocks.append(block)
        else:
            i += 1
    return blocks


def extract_all_test_sections(tests_dir: str = TESTS_DIR_DEFAULT) -> List[Dict[str, List[str]]]:
    all_blocks: List[Dict[str, List[str]]] = []
    for fname in os.listdir(tests_dir):
        if fname.startswith('test_layer_') and fname.endswith('.py'):
            path = os.path.join(tests_dir, fname)
            blocks = extract_test_sections_from_file(path)
            # Normalize left-hand variable names to filter only relevant ones and maintain order
            norm_blocks: List[Dict[str, List[str]]] = []
            for block in blocks:
                new_block = {'feedforward': [], 'backpropagation': [], '__meta__': block.get('__meta__', {'file': path, 'index': 0})}
                for section in ('feedforward', 'backpropagation'):
                    for line in block[section]:
                        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
                        if not m:
                            continue
                        var = m.group(1)
                        if var in EQ_VARS[section]:
                            new_block[section].append(line)
                norm_blocks.append(new_block)
            all_blocks.extend(norm_blocks)
    return all_blocks
