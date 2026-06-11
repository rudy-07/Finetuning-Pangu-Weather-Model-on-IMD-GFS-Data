"""Convert Pangu ONNX weights to a PyTorch checkpoint.

Fixes applied vs the original pseudocode:
  1. 2D (Linear) weights are ALWAYS transposed: ONNX MatMul stores
     weights as (in_features, out_features) while PyTorch nn.Linear
     stores them as (out_features, in_features).  The previous version
     short-circuited on `w.shape == param.shape` which silently skipped
     the transpose for *square* linear layers (attention.linear2, etc.).
  2. earth_specific_bias is expanded from the ONNX lookup-table form
     (3312, type_of_windows, heads) to the PyTorch direct-use form
     (1, type_of_windows, heads, 144, 144) via position_index.
  3. Comprehensive logging so every weight assignment is visible.
"""

import argparse
import os
import sys
from typing import Dict, Tuple

import numpy as np
import onnx
import onnx.numpy_helper as np_helper
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.pangu_model import PanguModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_lookup_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"keys_all.csv not found: {path}")
    table = pd.read_csv(path)
    if "torch_name" not in table or "onnx_name" not in table:
        raise ValueError("keys_all.csv must contain torch_name and onnx_name columns")
    return table


def _load_onnx_weights(onnx_path: str) -> Dict[str, np.ndarray]:
    model = onnx.load(onnx_path)
    weights: Dict[str, np.ndarray] = {}
    for initializer in model.graph.initializer:
        weights[initializer.name] = np_helper.to_array(initializer)

    # Also capture Constant nodes when present.
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                weights[node.output[0]] = np_helper.to_array(attr.t)

    return weights


# ---------------------------------------------------------------------------
# Earth-Specific Bias expansion (lookup-table → direct form)
# ---------------------------------------------------------------------------

def _build_position_index() -> torch.Tensor:
    """Reconstruct the position_index exactly as EarthAttention3D._construct_index() does.

    This maps the compact lookup-table bias of shape
        (3312, type_of_windows, heads)
    into the expanded attention-bias of shape
        (144, 144, type_of_windows, heads)
    via indexing.
    """
    window_size = (2, 6, 12)

    coords_zi = torch.arange(window_size[0])
    coords_zj = -torch.arange(window_size[0]) * window_size[0]

    coords_hi = torch.arange(window_size[1])
    coords_hj = -torch.arange(window_size[1]) * window_size[1]

    coords_w = torch.arange(window_size[2])

    coords_1 = torch.stack(torch.meshgrid(coords_zi, coords_hi, coords_w, indexing="ij"))
    coords_2 = torch.stack(torch.meshgrid(coords_zj, coords_hj, coords_w, indexing="ij"))
    coords_flatten_1 = torch.flatten(coords_1, start_dim=1)
    coords_flatten_2 = torch.flatten(coords_2, start_dim=1)

    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = torch.permute(coords, (1, 2, 0))

    coords[:, :, 2] += window_size[2] - 1
    coords[:, :, 1] *= 2 * window_size[2] - 1
    coords[:, :, 0] *= (2 * window_size[2] - 1) * window_size[1] * window_size[1]

    position_index = torch.sum(coords, dim=-1)
    position_index = torch.flatten(position_index)  # (20736,)
    return position_index


def _expand_earth_specific_bias(w: torch.Tensor, param_shape: Tuple[int, ...]) -> torch.Tensor:
    """Expand lookup-table bias to direct-use form.

    ONNX form:    (3312, type_of_windows, heads)
    PyTorch form: (1, type_of_windows, heads, 144, 144)
    """
    window_volume = 2 * 6 * 12  # = 144
    n_positions = (2 * 12 - 1) * 6 * 6 * 2 * 2  # = 3312

    if w.ndim == 3 and w.shape[0] == n_positions:
        # w is (3312, type_of_windows, heads) — lookup-table form
        position_index = _build_position_index()  # (20736,)
        expanded = w[position_index]  # (20736, type_of_windows, heads)
        expanded = expanded.view(window_volume, window_volume,
                                 w.shape[1], w.shape[2])  # (144, 144, tw, h)
        expanded = expanded.permute(2, 3, 0, 1)  # (tw, h, 144, 144)
        expanded = expanded.unsqueeze(0)  # (1, tw, h, 144, 144)
        assert expanded.shape == param_shape, (
            f"Expanded bias shape {tuple(expanded.shape)} != param shape {param_shape}"
        )
        return expanded

    if w.ndim == 4 and (1,) + tuple(w.shape) == param_shape:
        # w is (type_of_windows, heads, 144, 144) — just needs unsqueeze
        return w.unsqueeze(0)

    if tuple(w.shape) == param_shape:
        # Already in direct form
        return w

    raise ValueError(
        f"Cannot convert earth_specific_bias: ONNX shape {tuple(w.shape)} "
        f"→ PyTorch shape {param_shape}"
    )


# ---------------------------------------------------------------------------
# Weight assignment
# ---------------------------------------------------------------------------

def _assign_weight(
    param: torch.nn.Parameter,
    w: torch.Tensor,
    torch_name: str,
    onnx_name: str,
) -> torch.Tensor:
    """Transform an ONNX weight tensor to match the PyTorch parameter shape.

    Key rule for 2D (Linear) weights:
        ONNX MatMul stores W as (in_features, out_features).
        PyTorch nn.Linear stores W as (out_features, in_features).
        → ALWAYS transpose.  The previous version had a bug where it skipped
          the transpose for square matrices (shape == shape was True).
    """
    # --- earth_specific_bias (special case) ---
    if "earth_specific_bias" in torch_name:
        return _expand_earth_specific_bias(w, tuple(param.shape))

    # --- 2D: Linear weights — ALWAYS transpose ---
    if param.ndim == 2:
        wt = w.T
        if wt.shape == param.shape:
            return wt
        # This should not happen for correctly exported ONNX models,
        # but we keep it as a safety check.
        raise ValueError(
            f"Shape mismatch for {onnx_name} (torch: {torch_name}): "
            f"ONNX {tuple(w.shape)}, transposed {tuple(wt.shape)}, "
            f"expected {tuple(param.shape)}"
        )

    # --- 5D with missing batch dim (e.g., Conv3d-like) ---
    if param.ndim == 5 and w.ndim == 4 and w.shape == param.shape[1:]:
        return w.unsqueeze(0)

    # --- Everything else: direct shape match ---
    if w.shape != param.shape:
        raise ValueError(
            f"Shape mismatch for {onnx_name} (torch: {torch_name}): "
            f"ONNX {tuple(w.shape)} vs expected {tuple(param.shape)}"
        )
    return w


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(onnx_path: str, keys_path: str, output_path: str, strict: bool = True) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PanguModel(device=device).to(device)
    lookup = _load_lookup_table(keys_path)
    onnx_weights = _load_onnx_weights(onnx_path)

    print(f"ONNX weights loaded: {len(onnx_weights)} tensors")
    print(f"PyTorch model parameters: {sum(1 for _ in model.named_parameters())}")
    print(f"Lookup table rows: {len(lookup)}")
    print()

    missing = []
    loaded = []
    failed = []

    with torch.no_grad():
        for name, param in model.named_parameters():
            row = lookup[lookup["torch_name"] == name]
            if row.empty:
                missing.append(f"  NO CSV ROW  → {name}")
                continue

            onnx_name = row["onnx_name"].values[0]
            if not isinstance(onnx_name, str) or onnx_name not in onnx_weights:
                missing.append(f"  NO ONNX WGT → {name}  (onnx_name={onnx_name})")
                continue

            w = torch.tensor(onnx_weights[onnx_name], device=device)
            try:
                w = _assign_weight(param, w, name, onnx_name)
                param.copy_(w)
                loaded.append(name)
                print(f"  ✓ {name:<75s}  {str(tuple(param.shape)):>25s}  ← {onnx_name}")
            except ValueError as exc:
                failed.append(f"  SHAPE ERROR → {name}: {exc}")

    # ---- Summary ----
    print()
    print("=" * 80)
    print(f"SUMMARY: {len(loaded)} loaded, {len(missing)} missing, {len(failed)} failed")
    print("=" * 80)

    if missing:
        print("\nMISSING (not loaded — will use random init!):")
        for m in missing:
            print(m)

    if failed:
        print("\nFAILED (shape errors):")
        for f in failed:
            print(f)

    if missing or failed:
        msg = (
            f"{len(missing)} missing + {len(failed)} failed out of "
            f"{len(loaded) + len(missing) + len(failed)} parameters"
        )
        if strict:
            raise ValueError(msg)
        print(f"\nWARNING: {msg}")
    else:
        print("\nAll parameters loaded successfully!")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict()}, output_path)
    print(f"\nSaved torch checkpoint: {output_path}")


def main() -> None:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description="Convert Pangu ONNX weights to PyTorch format")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--keys", default=os.path.join(repo_root, "keys_all.csv"), help="Path to keys_all.csv")
    parser.add_argument("--out", default=None, help="Output .pth path (default: <onnx>_torch.pth)")
    parser.add_argument("--strict", action="store_true", help="Fail if any mapping/weight is missing")
    args = parser.parse_args()

    output_path = args.out or (os.path.splitext(args.onnx)[0] + "_torch.pth")
    convert(args.onnx, args.keys, output_path, strict=args.strict)


if __name__ == "__main__":
    main()
