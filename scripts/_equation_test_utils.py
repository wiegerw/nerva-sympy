#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import os
import re
import ast
import logging
from typing import Dict, List, Tuple, Optional, Callable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR_DEFAULT = os.path.join(ROOT, 'tests')

# Variables considered in the SymPy test blocks
EQ_VARS = {
    'feedforward': {'R', 'Sigma', 'inv_sqrt_Sigma', 'Z', 'Y'},
    'backpropagation': {'DZ', 'DW', 'Db', 'DX', 'Dbeta', 'Dgamma', 'Dal', 'Dar', 'Dtl', 'Dtr'},
}

_ws_re = re.compile(r"\s+")

# Base normalizer used for parsing SymPy tests (textual blocks)
# Additional normalizers and backend specifics can be layered on top via norm_expr_backend.

def norm_expr_tests(s: str) -> str:
    s = s.split('#', 1)[0]
    s = s.strip()
    s = _ws_re.sub(' ', s)
    s = s.replace(' .T', '.T')
    s = re.sub(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\((.*)\)$", r"\1 = \2", s)
    return s


def norm_expr_backend(replace_matmul_with_mul: bool = False,
                      extra_normalizers: Optional[List[Callable[[str], str]]] = None) -> Callable[[str], str]:
    """Create a normalization function for backend source expressions.
    - replace_matmul_with_mul: if True, replace '@' with '*'
    - extra_normalizers: list of callables(text)->text to apply after base normalization
    """
    def _f(s: str) -> str:
        s = s.split('#', 1)[0]
        s = s.strip()
        s = _ws_re.sub(' ', s)
        s = s.replace(' .T', '.T')
        if extra_normalizers:
            for g in extra_normalizers:
                s = g(s)
        if replace_matmul_with_mul:
            s = s.replace('@', '*')
        s = re.sub(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\((.*)\)$", r"\1 = \2", s)
        return s
    return _f


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
                    block['feedforward'].append(norm_expr_tests(line))
                i += 1
            # advance to backpropagation
            while i < n and not lines[i].strip().startswith('# backpropagation'):
                i += 1
            if i < n and lines[i].strip().startswith('# backpropagation'):
                i += 1
                while i < n and lines[i].strip() and not lines[i].lstrip().startswith('#'):
                    line = lines[i]
                    if '=' in line:
                        block['backpropagation'].append(norm_expr_tests(line))
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


# ---------------------- Backend/source extraction helpers ----------------------

LAYER_KEYS = {
    'LinearLayer': 'LinearLayer',
    'ActivationLayer': 'ActivationLayer',
    'SReLULayer': 'SReLULayer',
    'SoftmaxLayer': 'SoftmaxLayer',
    'LogSoftmaxLayer': 'LogSoftmaxLayer',
    'BatchNormalizationLayer': 'BatchNormalizationLayer',
}


def normalize_tf_transpose_calls(text: str) -> str:
    prefixes = ('tf.transpose(', 'tr.transpose(')
    changed = True
    while changed:
        changed = False
        for pref in prefixes:
            i = 0
            while True:
                idx = text.find(pref, i)
                if idx == -1:
                    break
                j = idx + len(pref)
                depth = 1
                while j < len(text) and depth > 0:
                    c = text[j]
                    if c == '(':
                        depth += 1
                    elif c == ')':
                        depth -= 1
                    j += 1
                if depth != 0:
                    i = idx + 1
                    continue
                inner = text[idx + len(pref): j - 1]
                inner_stripped = inner.strip()
                repl = f"{inner_stripped}.T"
                text = text[:idx] + repl + text[j:]
                changed = True
                i = idx + len(repl)
    return text


def extract_method_equations_from_def(src: str, func_def: Optional[ast.FunctionDef],
                                       normalize_expr: Callable[[str], str]) -> List[Tuple[str, str]]:
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
                line = normalize_expr(f"{var} = {seg}")
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


def resolve_method_equations(src: str, class_map: Dict[str, ast.ClassDef], cls_name: str, method_name: str,
                             normalize_expr: Callable[[str], str]) -> List[Tuple[str, str]]:
    cls = class_map.get(cls_name)
    if not cls:
        return []
    method = find_method_def(cls, method_name)
    eqs = extract_method_equations_from_def(src, method, normalize_expr) if method else []
    if cls_name == 'SReLULayer' and method_name == 'backpropagate':
        eqs = resolve_method_equations(src, class_map, 'ActivationLayer', 'backpropagate', normalize_expr) + eqs
        return eqs
    if not eqs:
        for base in get_base_names(cls):
            if base in class_map:
                eqs = resolve_method_equations(src, class_map, base, method_name, normalize_expr)
                if eqs:
                    break
    return eqs


def extract_layer_equations_generic(layers_file: str,
                                    replace_matmul_with_mul: bool = False,
                                    extra_normalizers: Optional[List[Callable[[str], str]]] = None) -> Dict[str, Dict[str, List[str]]]:
    with open(layers_file, 'r', encoding='utf-8') as f:
        src = f.read()
    tree = ast.parse(src)
    class_map = build_class_map(tree)

    normalize_expr = norm_expr_backend(replace_matmul_with_mul=replace_matmul_with_mul,
                                       extra_normalizers=extra_normalizers)

    layers: Dict[str, Dict[str, List[str]]] = {}
    for cls_name in LAYER_KEYS:
        node = class_map.get(cls_name)
        if not node:
            continue
        feed = resolve_method_equations(src, class_map, cls_name, 'feedforward', normalize_expr)
        back = resolve_method_equations(src, class_map, cls_name, 'backpropagate', normalize_expr)
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
