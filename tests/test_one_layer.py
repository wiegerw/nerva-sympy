#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import json
import os
import unittest
from pathlib import Path
from typing import Dict, Any

import numpy as np
import sympy as sp

from nerva_sympy.utilities import load_dict_from_npz
from nerva_sympy.layers import (
    LinearLayer,
    ActivationLayer,
    SoftmaxLayer,
    LogSoftmaxLayer,
    BatchNormalizationLayer,
    SReLULayer,
)
from nerva_sympy.activation_functions import (
    SReLUActivation,
    parse_activation,
)

from utilities import to_numpy

ESSENTIAL_ATOL = 1e-6
ESSENTIAL_RTOL = 1e-6
# Optional debug output controlled via ONE_LAYER_DEBUG environment variable.
# Values: "0" or unset = silent (default), "1" = basic, "2" = verbose tensors.
DEBUG_LEVEL = int(os.environ.get("ONE_LAYER_DEBUG", "0") or 0)

def _print_debug(msg: str):
    if DEBUG_LEVEL > 0:
        print(msg)


def np_to_sp_matrix(x: np.ndarray, force_row: bool = False) -> sp.Matrix:
    """Convert a numpy array to a SymPy Matrix, preserving 2D shapes.
    For 1D arrays, produce a 1xN row if force_row or if it's intended as bias.
    """
    x = np.array(x)
    if x.ndim == 0:
        return sp.Matrix([[float(x.item())]])
    if x.ndim == 1:
        if force_row:
            return sp.Matrix([x.tolist()])
        else:
            # default to column vector
            return sp.Matrix(x.tolist())
    else:
        return sp.Matrix(x.tolist())


def assert_close(name: str, a: sp.Matrix, b: np.ndarray, atol=ESSENTIAL_ATOL, rtol=ESSENTIAL_RTOL):
    A = to_numpy(a)
    B = np.array(b)
    # Harmonize shapes: allow (1,K) vs (K,) and (N,1) vs (N,)
    if A.shape != B.shape:
        if B.ndim == 1:
            if A.ndim == 2 and A.shape[0] == 1 and A.shape[1] == B.shape[0]:
                B = B.reshape(1, -1)
            elif A.ndim == 2 and A.shape[1] == 1 and A.shape[0] == B.shape[0]:
                B = B.reshape(-1, 1)
        # After reshape, if still mismatched, fail
    if A.shape != B.shape:
        _print_debug(f"ASSERT {name}: shape mismatch: {A.shape} vs {B.shape}")
        raise AssertionError(f"Shape mismatch for {name}: {A.shape} vs {B.shape}")
    close = np.allclose(A, B, atol=atol, rtol=rtol)
    if not close or DEBUG_LEVEL > 0:
        max_diff = float(np.max(np.abs(A - B))) if A.size and B.size else 0.0
        _print_debug(f"ASSERT {name}: shape={A.shape}, atol={atol}, rtol={rtol}, max|diff|={max_diff}")
        if DEBUG_LEVEL >= 2 and A.size <= 200:
            _print_debug(f"  A: {A}")
            _print_debug(f"  B: {B}")
    if not close:
        raise AssertionError(f"Mismatch in {name}")


def run_case(manifest_dir: Path, meta: Dict[str, Any]):
    tensors = load_dict_from_npz(str(manifest_dir / meta["file"]))

    # inputs are numeric; convert to SymPy matrices
    X = np_to_sp_matrix(tensors["X"])  # (N, D)

    _print_debug(f"CASE: type={meta.get('type')} file={meta.get('file')} D={meta.get('D')} K={meta.get('K')} activation={meta.get('activation_spec', meta.get('activation'))}")

    if meta["type"] == "Linear":
        D = meta["D"]; K = meta["K"]
        layer = LinearLayer(D, K)
        layer.W = np_to_sp_matrix(tensors["W"])  # (K, D)
        # bias should be row vector 1xK
        layer.b = np_to_sp_matrix(tensors["b"], force_row=True)
        Y = layer.feedforward(X)
        assert_close("Y", Y, tensors["Y"])
        DY = np_to_sp_matrix(tensors["DY"])  # (N, K)
        layer.backpropagate(Y, DY)
        assert_close("DX", layer.DX, tensors["DX"])  # (N, D)
        assert_close("DW", layer.DW, tensors["DW"])  # (K, D)
        assert_close("Db", layer.Db, tensors["Db"])  # (1, K)
        if "optimizer_spec" in meta:
            layer.set_optimizer(meta["optimizer_spec"])
            layer.optimize(meta.get("lr", 0.1))
            assert_close("W_opt", layer.W, tensors["W_opt"])
            assert_close("b_opt", layer.b, tensors["b_opt"])  # (1, K)
        return

    if meta["type"] == "Activation":
        D = meta["D"]; K = meta["K"]
        act = parse_activation(meta["activation_spec"])
        layer = ActivationLayer(D, K, act)
        layer.W = np_to_sp_matrix(tensors["W"])  # (K, D)
        layer.b = np_to_sp_matrix(tensors["b"], force_row=True)
        Y = layer.feedforward(X)
        assert_close("Y", Y, tensors["Y"])
        DY = np_to_sp_matrix(tensors["DY"])  # (N, K)
        layer.backpropagate(Y, DY)
        assert_close("DX", layer.DX, tensors["DX"])  # (N, D)
        assert_close("DW", layer.DW, tensors["DW"])  # (K, D)
        assert_close("Db", layer.Db, tensors["Db"])  # (1, K)
        # optimize verification
        if "optimizer_spec" in meta:
            layer.set_optimizer(meta["optimizer_spec"])
            layer.optimize(meta.get("lr", 0.1))
            assert_close("W_opt", layer.W, tensors.get("W_opt", tensors["W"]))
            assert_close("b_opt", layer.b, tensors.get("b_opt", tensors["b"]))
        return

    if meta["type"] == "SReLU":
        D = meta["D"]; K = meta["K"]
        act = parse_activation(meta["activation_spec"])
        layer = SReLULayer(D, K, act)
        layer.W = np_to_sp_matrix(tensors["W"])  # (K, D)
        layer.b = np_to_sp_matrix(tensors["b"], force_row=True)
        Y = layer.feedforward(X)
        assert_close("Y", Y, tensors["Y"])
        DY = np_to_sp_matrix(tensors["DY"])  # (N, K)
        layer.backpropagate(Y, DY)
        assert_close("DX", layer.DX, tensors["DX"])  # (N, D)
        assert_close("DW", layer.DW, tensors["DW"])  # (K, D)
        assert_close("Db", layer.Db, tensors["Db"])  # (1, K)
        assert_close("act.Dx", act.Dx, tensors["act_Dx"])
        # optimize verification
        if "optimizer_spec" in meta:
            layer.set_optimizer(meta["optimizer_spec"])
            layer.optimize(meta.get("lr", 0.1))
            assert_close("W_opt", layer.W, tensors.get("W_opt", tensors["W"]))
            assert_close("b_opt", layer.b, tensors.get("b_opt", tensors["b"]))
            assert_close("act_x_opt", layer.act.x, tensors["act_x_opt"])
        return

    if meta["type"] == "Softmax":
        D = meta["D"]; K = meta["K"]
        layer = SoftmaxLayer(D, K)
        layer.W = np_to_sp_matrix(tensors["W"])  # (K, D)
        layer.b = np_to_sp_matrix(tensors["b"], force_row=True)
        Y = layer.feedforward(X)
        assert_close("Y", Y, tensors["Y"])
        DY = np_to_sp_matrix(tensors["DY"])  # (N, K)
        layer.backpropagate(Y, DY)
        assert_close("DX", layer.DX, tensors["DX"])  # (N, D)
        assert_close("DW", layer.DW, tensors["DW"])  # (K, D)
        assert_close("Db", layer.Db, tensors["Db"])  # (1, K)
        if "optimizer_spec" in meta:
            layer.set_optimizer(meta["optimizer_spec"])
            layer.optimize(meta.get("lr", 0.1))
            assert_close("W_opt", layer.W, tensors["W_opt"])  # (K, D)
            assert_close("b_opt", layer.b, tensors["b_opt"])  # (1, K)
        return

    if meta["type"] == "LogSoftmax":
        D = meta["D"]; K = meta["K"]
        layer = LogSoftmaxLayer(D, K)
        layer.W = np_to_sp_matrix(tensors["W"])  # (K, D)
        layer.b = np_to_sp_matrix(tensors["b"], force_row=True)
        Y = layer.feedforward(X)
        assert_close("Y", Y, tensors["Y"])
        DY = np_to_sp_matrix(tensors["DY"])  # (N, K)
        layer.backpropagate(Y, DY)
        assert_close("DX", layer.DX, tensors["DX"])  # (N, D)
        assert_close("DW", layer.DW, tensors["DW"])  # (K, D)
        assert_close("Db", layer.Db, tensors["Db"])  # (1, K)
        if "optimizer_spec" in meta:
            layer.set_optimizer(meta["optimizer_spec"])
            layer.optimize(meta.get("lr", 0.1))
            assert_close("W_opt", layer.W, tensors["W_opt"])  # (K, D)
            assert_close("b_opt", layer.b, tensors["b_opt"])  # (1, K)
        return

    if meta["type"] == "BatchNormalization":
        D = meta["D"]
        layer = BatchNormalizationLayer(D)
        layer.gamma[:, :] = np_to_sp_matrix(tensors["gamma"], force_row=True)  # (1, D)
        layer.beta[:, :] = np_to_sp_matrix(tensors["beta"], force_row=True)    # (1, D)
        Y = layer.feedforward(X)
        assert_close("Y", Y, tensors["Y"])
        DY = np_to_sp_matrix(tensors["DY"])  # (N, D)
        layer.backpropagate(Y, DY)
        assert_close("DX", layer.DX, tensors["DX"])
        assert_close("Dgamma", layer.Dgamma, tensors["Dgamma"])
        assert_close("Dbeta", layer.Dbeta, tensors["Dbeta"])
        if "optimizer_spec" in meta:
            layer.set_optimizer(meta["optimizer_spec"])
            layer.optimize(meta.get("lr", 0.1))
            assert_close("gamma_opt", layer.gamma, tensors["gamma_opt"])
            assert_close("beta_opt", layer.beta, tensors["beta_opt"])
        return

    raise ValueError(f"Unknown case type: {meta['type']}")


class TestOneLayer(unittest.TestCase):
    def test_cases(self):
        # Read cases from the default directory under tests/one_layer_cases
        out_dir = Path(__file__).parent / "one_layer_cases"
        manifest_path = out_dir / "manifest.json"
        if not manifest_path.exists():
            self.skipTest(f"No one_layer_cases manifest found in '{manifest_path}'")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(manifest), 1, "No cases found in manifest")

        for meta in manifest:
            with self.subTest(name=meta.get("name", meta.get("file", "unknown"))):
                run_case(out_dir, meta)


if __name__ == '__main__':
    unittest.main()
