#!/usr/bin/env python3

import json
import re
import sys
from typing import Dict, List, Tuple


def normalize_layer_name(name: str) -> str:
    """Convert LaTeX paragraph name to PascalCase layer key."""
    name = name.strip().lower().replace(" layer", "")
    parts = re.split(r"[\s_-]+", name)
    return "".join(part.capitalize() for part in parts) + "Layer"


def extract_latex_equations(tex_path: str) -> Dict[str, Dict[str, List[str]]]:
    """Extract layer equations from LaTeX file."""
    with open(tex_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Isolate the Layer equations section
    section_match = re.search(r"\\section\{Layer equations}.*?(?=\\subsection|\Z)", content, re.DOTALL)
    if not section_match:
        print("❌ Could not find 'Layer equations' section.")
        return {}

    section = section_match.group(0)
    paragraphs = re.split(r"\\paragraph\{(.*?)}", section)[1:]  # Skip preamble

    layer_blocks = {}
    for i in range(0, len(paragraphs), 2):
        raw_name = paragraphs[i]
        body = paragraphs[i + 1]
        layer = normalize_layer_name(raw_name)

        # Extract lstlisting blocks
        listings = re.findall(r"\\begin\{lstlisting}(.*?)\\end\{lstlisting}", body, re.DOTALL)
        if len(listings) != 2:
            print(f"⚠️ Skipping {layer}: expected 2 lstlisting blocks, found {len(listings)}")
            continue

        feedforward = [line.strip() for line in listings[0].strip().splitlines() if line.strip()]
        backpropagation = [line.strip() for line in listings[1].strip().splitlines() if line.strip()]
        layer_blocks[layer] = {
            "feedforward": feedforward,
            "backpropagation": backpropagation
        }

    return layer_blocks


def compare_equations(json_data: Dict[str, List[Dict[str, List[str]]]],
                      tex_data: Dict[str, Dict[str, List[str]]]) -> None:
    """Compare equations and print report."""
    print("🔍 Comparing LaTeX equations with JSON blocks...\n")
    for tex_layer, tex_eqs in tex_data.items():
        # Try to find matching JSON layer
        json_layer = next((key for key in json_data if key.lower() == tex_layer.lower()), None)
        if not json_layer:
            print(f"❌ Layer '{tex_layer}' not found in JSON")
            continue

        # Compare against all blocks for that layer
        matched = False
        for block in json_data[json_layer]:
            if block["feedforward"] == tex_eqs["feedforward"] and block["backpropagation"] == tex_eqs["backpropagation"]:
                print(f"✅ Layer '{tex_layer}' matches JSON layer '{json_layer}'")
                matched = True
                break

        if not matched:
            print(f"❌ Layer '{tex_layer}' does not match JSON layer '{json_layer}'")
            print("  Differences:")
            print("  → Feedforward:")
            print("    JSON:", json_data[json_layer][0]["feedforward"])
            print("    TeX :", tex_eqs["feedforward"])
            print("  → Backpropagation:")
            print("    JSON:", json_data[json_layer][0]["backpropagation"])
            print("    TeX :", tex_eqs["backpropagation"])
        print()


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_layer_equations.py <equations.json> <layers.tex>")
        sys.exit(1)

    json_path = sys.argv[1]
    tex_path = sys.argv[2]

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    tex_data = extract_latex_equations(tex_path)
    compare_equations(json_data, tex_data)


if __name__ == "__main__":
    main()
