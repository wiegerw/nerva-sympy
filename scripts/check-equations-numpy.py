#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import ast
import os
import re
import argparse
import logging
from typing import Dict, List, Tuple, Optional

from _equation_test_utils import extract_all_test_sections, EQ_VARS

SYMPY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIBLING_ROOT = os.path.dirname(SYMPY_ROOT)
LAYERS_FILE_DEFAULT = os.path.join(SIBLING_ROOT, 'nerva-numpy', 'src', 'nerva_numpy', 'layers.py')
TESTS_DIR_DEFAULT = os.path.join(SYMPY_ROOT, 'tests')

LAYER_KEYS = {
    'LinearLayer': 'LinearLayer',
    'ActivationLayer': 'ActivationLayer',
    'SReLULayer': 'SReLULayer',
    'SoftmaxLayer': 'SoftmaxLayer',
    'LogSoftmaxLayer': 'LogSoftmaxLayer',
    'BatchNormalizationLayer': 'BatchNormalizationLayer',
}

_ws_re = re.compile(r"\s+")

def norm_expr(s: str) -> str:
    s = s.split('#', 1)[0]
    s = s.strip()
    s = _ws_re.sub(' ', s)
    s = s.replace(' .T', '.T')
    s = s.replace('@', '*')
    s = re.sub(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\((.*)\)$", r"\1 = \2", s)
    return s


def extract_method_equations_from_def(src: str, func_def: Optional[ast.FunctionDef]) -> List[Tuple[str, str]]:
    ordered: List[Tuple[str, str]] = []
    if func_def is None:
        return ordered
    for stmt in func_def.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                var = target.id
            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                var = target.attr
            else:
                var = None
            if var and var in (EQ_VARS['feedforward'] | EQ_VARS['backpropagation']):
                seg = ast.get_source_segment(src, stmt.value)
                if seg is None:
                    seg = ast.unparse(stmt.value) if hasattr(ast, 'unparse') else ''
                line = norm_expr(f"{var} = {seg}")
                if re.match(rf"^\s*{re.escape(var)}\s*=\s*{re.escape(var)}\s*$", line):
                    continue
                ordered.append((var, line))
    return ordered


def find_method_def(class_node: ast.ClassDef, method_name: str) -> ast.FunctionDef | None:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    return None


def build_class_map(tree: ast.Module) -> Dict[str, ast.ClassDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def get_base_names(cls: ast.ClassDef) -> List[str]:
    names = []
    for b in cls.bases:
        if isinstance(b, ast.Name):
            names.append(b.id)
        elif isinstance(b, ast.Attribute):
            names.append(b.attr)
    return names


def resolve_method_equations(src: str, class_map: Dict[str, ast.ClassDef], cls_name: str, method_name: str) -> List[Tuple[str, str]]:
    cls = class_map.get(cls_name)
    if not cls:
        return []
    method = find_method_def(cls, method_name)
    eqs = extract_method_equations_from_def(src, method) if method else []
    if cls_name == 'SReLULayer' and method_name == 'backpropagate':
        eqs = resolve_method_equations(src, class_map, 'ActivationLayer', 'backpropagate') + eqs
        return eqs
    if not eqs:
        for base in get_base_names(cls):
            if base in class_map:
                eqs = resolve_method_equations(src, class_map, base, method_name)
                if eqs:
                    break
    return eqs


def extract_layer_equations(layers_file: str) -> Dict[str, Dict[str, List[str]]]:
    with open(layers_file, 'r', encoding='utf-8') as f:
        src = f.read()
    tree = ast.parse(src)
    class_map = build_class_map(tree)

    layers: Dict[str, Dict[str, List[str]]] = {}
    for cls_name in LAYER_KEYS:
        node = class_map.get(cls_name)
        if not node:
            continue
        feed = resolve_method_equations(src, class_map, cls_name, 'feedforward')
        back = resolve_method_equations(src, class_map, cls_name, 'backpropagate')
        feed_eqs = [eq for var, eq in feed if var in EQ_VARS['feedforward']]
        back_eqs = [eq for var, eq in back if var in EQ_VARS['backpropagation']]
        layers[cls_name] = {
            'feedforward': feed_eqs,
            'backpropagation': back_eqs,
        }
    return layers


def block_distance(layer_eqs: Dict[str, List[str]], block: Dict[str, List[str]]) -> int:
    dist = 0
    for section in ('feedforward', 'backpropagation'):
        a = set(layer_eqs.get(section, []))
        b = set(block.get(section, []))
        dist += len(a.symmetric_difference(b))
    return dist


def print_block_equations(prefix: str, block: Dict[str, List[str]]) -> None:
    meta = block.get('__meta__', {})
    src = meta.get('file', '<unknown>')
    idx = meta.get('index', '?')
    print(f"{prefix} (from {src}, block #{idx}) feedforward:")
    for line in block.get('feedforward', []):
        print('  ' + line)
    print(f"{prefix} (from {src}, block #{idx}) backpropagation:")
    for line in block.get('backpropagation', []):
        print('  ' + line)


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Check consistency of nerva-numpy layer equations with SymPy tests')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity (use -vv for debug)')
    parser.add_argument('--layers-file', default=LAYERS_FILE_DEFAULT, help='Path to layers.py to inspect (default: ../nerva-numpy/src/nerva_numpy/layers.py)')
    parser.add_argument('--tests-dir', default=TESTS_DIR_DEFAULT, help='Path to SymPy tests directory (default: ./tests)')
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    layers = extract_layer_equations(args.layers_file)
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
