#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import os
import sys
import ast
import json
from typing import Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
LOSS_FILE = os.path.join(ROOT_DIR, "src", "nerva_sympy", "loss_functions.py")
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "loss-function-equations.json")
sys.path.insert(0, ROOT_DIR)


def extract_loss_expressions(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)

    losses: Dict[str, Dict[str, str]] = {}

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            name = node.name
            if not name.endswith("_loss") and not name.endswith("_loss_gradient"):
                continue
            if len(node.args.args) != 2:
                continue
            arg_names = [arg.arg for arg in node.args.args]
            if arg_names not in [["y", "t"], ["Y", "T"]]:
                continue

            # Extract expression from return statement
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    expr = ast.unparse(stmt.value).strip()
                    break
            else:
                continue

            # Normalize name
            base = name.replace("_gradient", "").replace("_loss", "")
            key = base.replace("_", " ").title().replace(" ", "")
            variant = "vector" if arg_names == ["y", "t"] else "matrix"
            kind = "gradient" if "gradient" in name else "value"
            losses.setdefault(key, {})[f"{kind}_{variant}"] = expr

    return losses


def print_loss_report(losses: Dict[str, Dict[str, str]]) -> None:
    for name, block in losses.items():
        print(f"{name}\n")
        print("# value (vector)")
        print(block.get("value_vector", "[missing]"))
        print("\n# gradient (vector)")
        print(block.get("gradient_vector", "[missing]"))
        print("\n# value (matrix)")
        print(block.get("value_matrix", "[missing]"))
        print("\n# gradient (matrix)")
        print(block.get("gradient_matrix", "[missing]"))
        print()


def export_to_json(losses: Dict[str, Dict[str, str]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(losses, f, indent=2, ensure_ascii=False)
    print(f"💾 Exported to {output_path}")


def main():
    print("🔍 Extracting loss functions from:", LOSS_FILE)
    losses = extract_loss_expressions(LOSS_FILE)
    print_loss_report(losses)
    export_to_json(losses, OUTPUT_JSON)
    print("✅ Done.")


if __name__ == "__main__":
    main()
