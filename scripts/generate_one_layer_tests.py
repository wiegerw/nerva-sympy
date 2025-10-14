#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""
Generate deterministic one-layer forward/backprop test cases for all supported
layers/activations in this repository. Results are saved as separate .npz files
with a manifest.json index for consumption by other implementations.

Each test case contains:
- X: input matrix (N x D)
- DY: upstream gradient dL/dY (N x K or N x D for BN)
- Layer parameters (W, b) or (gamma, beta) or activation params
- Y: layer output
- DX: gradient dL/dX
- Parameter gradients (DW, Db) or (Dgamma, Dbeta) or activation param grads

Usage:
    python tools/generate_one_layer_tests.py --out-dir out/one_layer_tests

The generator only depends on this repository (torch/numpy are transitive deps).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import torch

from nerva_torch.layers import (
    LinearLayer,
    ActivationLayer,
    SoftmaxLayer,
    LogSoftmaxLayer,
    BatchNormalizationLayer,
    SReLULayer,
)
from nerva_torch.activation_functions import (
    ReLUActivation,
    LeakyReLUActivation,
    AllReLUActivation,
    HyperbolicTangentActivation,
    SigmoidActivation,
    SReLUActivation,
)


def _tensor_to_list(x: torch.Tensor):
    return x.detach().cpu().numpy().tolist()


def _save_npz(path: Path, tensors: Dict[str, torch.Tensor]):
    import numpy as np
    numpy_data = {k: v.detach().cpu().numpy() for k, v in tensors.items()}
    np.savez_compressed(str(path), **numpy_data)


def _mk_linear_case(name: str, N: int, D: int, K: int, rng: torch.Generator, optimizer_spec: str, lr: float):
    layer = LinearLayer(D, K)
    X = torch.randn(N, D, generator=rng) * 0.5 + 0.1
    layer.W[:] = torch.randn(K, D, generator=rng) * 0.3
    layer.b[:] = torch.randn(K, generator=rng) * 0.1

    Y = layer.feedforward(X)
    DY = torch.randn(Y.shape, generator=rng, dtype=Y.dtype) * 0.2
    layer.backpropagate(Y, DY)

    case = {
        "name": name,
        "type": "Linear",
        "optimizer_spec": optimizer_spec,
        "lr": lr,
        "N": N,
        "D": D,
        "K": K,
        "X": X,
        "Y": Y,
        "DY": DY,
        "DX": layer.DX,
        "W": layer.W.clone(),
        "b": layer.b.clone(),
        "DW": layer.DW.clone(),
        "Db": layer.Db.clone(),
    }
    # optimize step (in-place on layer)
    layer.set_optimizer(optimizer_spec)
    layer.optimize(lr)
    case["W_opt"] = layer.W.clone()
    case["b_opt"] = layer.b.clone()
    return case


def _mk_activation_case(name: str, N: int, D: int, K: int, act, rng: torch.Generator, optimizer_spec: str, lr: float):
    layer = SReLULayer(D, K, act) if isinstance(act, SReLUActivation) else ActivationLayer(D, K, act)
    layer.set_optimizer(optimizer_spec)
    X = torch.randn(N, D, generator=rng) * 0.5 + 0.1
    layer.W[:] = torch.randn(K, D, generator=rng) * 0.3
    layer.b[:] = torch.randn(K, generator=rng) * 0.1

    Y = layer.feedforward(X)
    DY = torch.randn(Y.shape, generator=rng, dtype=Y.dtype) * 0.2
    layer.backpropagate(Y, DY)

    case = {
        "name": name,
        "type": "Activation",
        "optimizer_spec": optimizer_spec,
        "lr": lr,
        "activation_spec": str(act),
        "N": N,
        "D": D,
        "K": K,
        "X": X,
        "Y": Y,
        "DY": DY,
        "DX": layer.DX,
        "W": layer.W.clone(),
        "b": layer.b.clone(),
        "DW": layer.DW.clone(),
        "Db": layer.Db.clone(),
    }
    # SReLU has trainable activation params stored in act.x and act.Dx
    if isinstance(act, SReLUActivation):
        case["act_x"] = act.x.clone()
        case["act_Dx"] = act.Dx.clone()

    # optimize
    layer.optimize(lr)
    case["W_opt"] = layer.W.clone()
    case["b_opt"] = layer.b.clone()
    if isinstance(act, SReLUActivation):
        case["act_x_opt"] = act.x.clone()

    return case


def _mk_softmax_case(name: str, N: int, D: int, K: int, rng: torch.Generator, kind: str, optimizer_spec: str, lr: float):
    layer = SoftmaxLayer(D, K) if kind == "softmax" else LogSoftmaxLayer(D, K)
    X = torch.randn(N, D, generator=rng) * 0.5 + 0.1
    layer.W[:] = torch.randn(K, D, generator=rng) * 0.3
    layer.b[:] = torch.randn(K, generator=rng) * 0.1

    Y = layer.feedforward(X)
    DY = torch.randn(Y.shape, generator=rng, dtype=Y.dtype) * 0.2
    layer.backpropagate(Y, DY)

    case = {
        "name": name,
        "type": "Softmax" if kind == "softmax" else "LogSoftmax",
        "optimizer_spec": optimizer_spec,
        "lr": lr,
        "N": N,
        "D": D,
        "K": K,
        "X": X,
        "Y": Y,
        "DY": DY,
        "DX": layer.DX,
        "W": layer.W.clone(),
        "b": layer.b.clone(),
        "DW": layer.DW.clone(),
        "Db": layer.Db.clone(),
    }
    layer.set_optimizer(optimizer_spec)
    layer.optimize(lr)
    case["W_opt"] = layer.W.clone()
    case["b_opt"] = layer.b.clone()
    return case


def _mk_batchnorm_case(name: str, N: int, D: int, rng: torch.Generator, optimizer_spec: str, lr: float):
    layer = BatchNormalizationLayer(D)
    X = torch.randn(N, D, generator=rng) * 0.5 + 0.1
    # Set non-trivial gamma and beta
    layer.gamma[:] = torch.randn(D, generator=rng) * 0.4 + 1.0
    layer.beta[:] = torch.randn(D, generator=rng) * 0.2

    Y = layer.feedforward(X)
    DY = torch.randn(Y.shape, generator=rng, dtype=Y.dtype) * 0.2
    layer.backpropagate(Y, DY)

    case = {
        "name": name,
        "type": "BatchNormalization",
        "optimizer_spec": optimizer_spec,
        "lr": lr,
        "N": N,
        "D": D,
        "X": X,
        "Y": Y,
        "DY": DY,
        "DX": layer.DX,
        "gamma": layer.gamma.clone(),
        "beta": layer.beta.clone(),
        "Dgamma": layer.Dgamma.clone(),
        "Dbeta": layer.Dbeta.clone(),
    }
    layer.set_optimizer(optimizer_spec)
    layer.optimize(lr)
    case["gamma_opt"] = layer.gamma.clone()
    case["beta_opt"] = layer.beta.clone()
    return case


def _materialize_case(dirpath: Path, index: List[Dict[str, Any]], i: int, case: Dict[str, Any]):
    # Write tensors to npz, metadata to manifest (JSON)
    tensor_keys = [
        "X", "Y", "DY", "DX",
        "W", "b", "DW", "Db",
        "W_opt", "b_opt",
        "gamma", "beta", "Dgamma", "Dbeta",
        "gamma_opt", "beta_opt",
        "act_x", "act_Dx", "act_x_opt",
    ]
    tensors = {k: v for k, v in case.items() if k in tensor_keys and v is not None}

    out_npz = dirpath / f"case_{i:03d}.npz"
    _save_npz(out_npz, tensors)

    # Prepare manifest entry
    meta = {k: v for k, v in case.items() if k not in tensors}
    meta["file"] = out_npz.name
    # For readability, also include shapes and some scalars
    if "W" in tensors:
        meta["W_shape"] = list(tensors["W"].shape)
        meta["b_shape"] = list(tensors["b"].shape)
    if "gamma" in tensors:
        meta["gamma_shape"] = list(tensors["gamma"].shape)
        meta["beta_shape"] = list(tensors["beta"].shape)
    index.append(meta)


def main():
    import nerva_torch

    parser = argparse.ArgumentParser()
    # Default output directory inside the repository under tests/one_layer_cases
    repo_root = Path(__file__).resolve().parents[1]
    default_out_dir = repo_root / "tests" / "one_layer_cases"
    parser.add_argument("--out-dir", type=str, default=str(default_out_dir), help="Output directory for generated testcases")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for determinism")
    parser.add_argument("--Ns", type=str, default="7", help="Comma-separated batch sizes (N)")
    parser.add_argument("--Ds", type=str, default="4,5", help="Comma-separated input dims (D)")
    parser.add_argument("--Ks", type=str, default="3,4", help="Comma-separated output dims (K)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate used for the optimize step")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = torch.Generator().manual_seed(args.seed)

    # We set the constant for numerical stability to 0, in order to be consistent with nerva-sympy
    nerva_torch.matrix_operations.epsilon = 0

    Ns = [int(x) for x in str(args.Ns).split(',') if x]
    Ds = [int(x) for x in str(args.Ds).split(',') if x]
    Ks = [int(x) for x in str(args.Ks).split(',') if x]

    index: List[Dict[str, Any]] = []
    i = 0

    # Activations (ActivationLayer)
    activations = [
        (ReLUActivation(), "ReLU"),
        (LeakyReLUActivation(0.2), "LeakyReLU_0p2"),
        (AllReLUActivation(0.7), "AllReLU_0p7"),
        (HyperbolicTangentActivation(), "Tanh"),
        (SigmoidActivation(), "Sigmoid"),
        (SReLUActivation(al=0.1, tl=-0.2, ar=0.3, tr=0.5), "SReLU"),
    ]

    # Optimizers used for the optimize step
    optimizers = [
        "GradientDescent",
        "Momentum(mu=0.9)",
        "Nesterov(mu=0.9)",
    ]
    # Linear cases for all combinations of N, D, K
    for N in Ns:
        for D in Ds:
            for K in Ks:
                for opt in optimizers:
                    case = _mk_linear_case(f"Linear_D{D}_K{K}_{opt}", N=N, D=D, K=K, rng=rng, optimizer_spec=opt, lr=args.lr)
                    _materialize_case(out_dir, index, i, case); i += 1

    # ActivationLayer cases for all combinations
    for act, tag in activations:
        for N in Ns:
            for D in Ds:
                for K in Ks:
                    for opt in optimizers:
                        case = _mk_activation_case(f"Activation_{tag}_D{D}_K{K}_{opt}", N=N, D=D, K=K, act=act, rng=rng, optimizer_spec=opt, lr=args.lr)
                        _materialize_case(out_dir, index, i, case); i += 1

    # Softmax and LogSoftmax for all combinations
    for N in Ns:
        for D in Ds:
            for K in Ks:
                for opt in optimizers:
                    case = _mk_softmax_case(f"Softmax_D{D}_K{K}_{opt}", N=N, D=D, K=K, rng=rng, kind="softmax", optimizer_spec=opt, lr=args.lr)
                    _materialize_case(out_dir, index, i, case); i += 1
                    case = _mk_softmax_case(f"LogSoftmax_D{D}_K{K}_{opt}", N=N, D=D, K=K, rng=rng, kind="logsoftmax", optimizer_spec=opt, lr=args.lr)
                    _materialize_case(out_dir, index, i, case); i += 1

    # BatchNormalization for N and D combinations (no K)
    for N in Ns:
        for D in Ds:
            for opt in optimizers:
                case = _mk_batchnorm_case(f"BatchNorm_D{D}_{opt}", N=N, D=D, rng=rng, optimizer_spec=opt, lr=args.lr)
                _materialize_case(out_dir, index, i, case); i += 1

    # Write manifest
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Wrote {len(index)} testcases to {out_dir}")
    print(f"Manifest: {manifest_path}")

    # Mirror the generated cases into sibling repositories under their tests/ dirs
    parent = repo_root.parent
    targets = [
        "nerva-jax",
        "nerva-numpy",
        "nerva-torch",
        "nerva-tensorflow",
        "nerva-rowwise",
        "nerva-colwise",
        "nerva-sympy",
    ]

    def _copy_tree(src: Path, dst: Path):
        import shutil
        dst.mkdir(parents=True, exist_ok=True)
        # Remove existing files in dst/one_layer_cases to keep it in sync
        for p in dst.iterdir():
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
        # Copy tree
        for p in src.iterdir():
            sp = p
            dp = dst / p.name
            if p.is_dir():
                shutil.copytree(sp, dp)
            else:
                shutil.copy2(sp, dp)

    for t in targets:
        repo = parent / t
        dst = repo / "tests" / "one_layer_cases"
        try:
            if not (repo.exists() and repo.is_dir()):
                print(f"Skip sync: {repo} not found")
                continue
            # Avoid syncing to self (would clear the just-written directory)
            if dst.resolve() == out_dir.resolve():
                print(f"Skip sync to self: {dst}")
                continue
            _copy_tree(out_dir, dst)
            print(f"Synced {out_dir} -> {dst}")
        except Exception as e:
            print(f"Warning: failed to sync to {dst}: {e}")


if __name__ == "__main__":
    main()
