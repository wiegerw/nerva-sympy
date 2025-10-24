#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import os
import sys
import json
import re
from typing import Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
TESTS_DIR = os.path.join(ROOT_DIR, "tests")
sys.path.insert(0, ROOT_DIR)


def normalize_layer_key(raw_name: str) -> str:
    """
    Normalize a test function name like 'test_yeh_batch_normalization_layer'
    into a unique layer key like 'YehBatchNormalizationLayer'.
    """
    name = raw_name.strip().lower()

    # Remove common prefixes and suffixes
    for prefix in ("test_", "_test_", "testlayer_", "_testlayer_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    for suffix in ("_layer", "_layers"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]

    # Convert to PascalCase
    parts = re.split(r"[_\s]+", name)
    return "".join(part.capitalize() for part in parts) + "Layer"


def extract_contiguous_equations(lines: List[str], start_marker: str) -> List[str]:
    """Extract contiguous assignment lines following a section marker like '# feedforward'."""
    equations = []
    collecting = False
    for line in lines:
        if start_marker in line:
            collecting = True
            continue
        if collecting:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                break  # Stop at blank line or new comment
            # Remove trailing comment
            code = line.split("#", 1)[0].strip()
            # Only include assignment lines
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", code):
                equations.append(code)
            else:
                break  # Stop if it's not an assignment
    return equations


def extract_equation_blocks_from_file(path: str) -> Dict[str, List[Dict[str, List[str]]]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    blocks_by_layer: Dict[str, List[List[str]]] = {}
    current_layer = None
    current_block = []
    for line in lines:
        m = re.match(r"^\s*def _?test_([a-zA-Z0-9_]+)_layer", line)
        if m:
            if current_layer and current_block:
                blocks_by_layer.setdefault(current_layer, []).append(current_block)
            raw_name = m.group(1)
            current_layer = normalize_layer_key(raw_name)
            current_block = [line]
        elif current_block is not None:
            current_block.append(line)
    if current_layer and current_block:
        blocks_by_layer.setdefault(current_layer, []).append(current_block)

    parsed: Dict[str, List[Dict[str, List[str]]]] = {}
    for layer, blocks in blocks_by_layer.items():
        for block_lines in blocks:
            feedforward = extract_contiguous_equations(block_lines, "# feedforward")
            backpropagation = extract_contiguous_equations(block_lines, "# backpropagation")
            if feedforward and backpropagation:
                parsed.setdefault(layer, []).append({
                    "feedforward": feedforward,
                    "backpropagation": backpropagation,
                    "__meta__": {"file": path}
                })
    return parsed


def collect_all_equation_blocks(tests_dir: str) -> Dict[str, List[Dict[str, List[str]]]]:
    all_blocks: Dict[str, List[Dict[str, List[str]]]] = {}
    for fname in os.listdir(tests_dir):
        if fname.startswith("test_layer_") and fname.endswith(".py"):
            path = os.path.join(tests_dir, fname)
            blocks = extract_equation_blocks_from_file(path)
            for layer, entries in blocks.items():
                all_blocks.setdefault(layer, []).extend(entries)
    return all_blocks


def print_equation_overview(blocks_by_layer: Dict[str, List[Dict[str, List[str]]]]) -> None:
    for layer, blocks in blocks_by_layer.items():
        for block in blocks:
            print(f"{layer}\n")
            print("# feedforward:")
            for line in block.get("feedforward", []):
                print(line)
            print("\n# backpropagation:")
            for line in block.get("backpropagation", []):
                print(line)
            print()  # extra newline between blocks


def export_to_json(blocks_by_layer: Dict[str, List[Dict[str, List[str]]]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(blocks_by_layer, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Exported to {output_path}")


def main():
    print("🔍 Extracting equation blocks from test_layer files...")
    blocks_by_layer = collect_all_equation_blocks(TESTS_DIR)

    print("\n📋 Printing overview of layer equations:")
    print_equation_overview(blocks_by_layer)

    output_path = os.path.join(SCRIPT_DIR, "layer_equations.json")
    export_to_json(blocks_by_layer, output_path)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
