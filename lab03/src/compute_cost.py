"""
Module for computing model parameters, hand-calculated FLOPs and MACs,
and profiling using torchinfo and thop.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torchinfo import summary
import thop


def count_transformer_encoder_layer(m: nn.Module, x: Tuple[torch.Tensor, ...], y: Any) -> None:
    """
    Custom OP hook for thop to accurately profile PyTorch's native nn.TransformerEncoderLayer.
    Computes exact MACs for:
      - Q, K, V linear projections: 3 * (S * d_model * d_model)
      - Attention matrix multiplication (QK^T): S * S * d_model
      - Context matrix multiplication (AV): S * S * d_model
      - Output linear projection: S * d_model * d_model
      - Feed-Forward Network (FFN): S * d_model * d_ff + S * d_ff * d_model
    """
    x_in = x[0]
    B, S, D = x_in.shape
    d_ff = m.linear1.out_features
    # Multi-Head Attention MACs
    mha_macs = 3 * S * D * D + 2 * S * S * D + S * D * D
    # Feed-Forward Network MACs
    ffn_macs = 2 * S * D * d_ff
    total_macs = B * (mha_macs + ffn_macs)
    m.total_ops += torch.DoubleTensor([int(total_macs)])


def hand_calculate_layer_costs() -> Dict[str, Dict[str, Any]]:
    """
    Computes theoretical parameters, MACs, and FLOPs for representative layers
    using Reddi Chapter 4 formulas:
      - Linear: FLOPs = 2 * in * out, MACs = in * out
      - Conv2d: FLOPs = 2 * K * K * C_in * C_out * H_out * W_out, MACs = K * K * C_in * C_out * H_out * W_out
      - Attention (QKV, Attention Scores, AV context, Out Proj, FFN, and Total Layer)
    """
    costs = {}

    # 1. Representative Linear Layer: MLP Layer 1 (784 -> 256)
    lin_in = 784
    lin_out = 256
    lin_params = lin_in * lin_out + lin_out  # 200,960
    lin_macs = lin_in * lin_out              # 200,704
    lin_flops = 2 * lin_macs                 # 401,408
    costs["Linear (MLP Layer 1: 784->256)"] = {
        "formula_params": "in * out + out",
        "params": lin_params,
        "formula_macs": "in * out",
        "macs": lin_macs,
        "formula_flops": "2 * in * out",
        "flops": lin_flops,
    }

    # 2. Representative Conv2d Layer: CNN Conv 2 (16 -> 32, K=3, H_out=14, W_out=14)
    c_in = 16
    c_out = 32
    k = 3
    h_out, w_out = 14, 14
    conv_params = c_in * c_out * k * k + c_out  # 4,640
    conv_macs = k * k * c_in * c_out * h_out * w_out  # 903,168
    conv_flops = 2 * conv_macs  # 1,806,336
    costs["Conv2d (CNN Conv 2: 16->32, 3x3)"] = {
        "formula_params": "K * K * C_in * C_out + C_out",
        "params": conv_params,
        "formula_macs": "K * K * C_in * C_out * H_out * W_out",
        "macs": conv_macs,
        "formula_flops": "2 * K * K * C_in * C_out * H_out * W_out",
        "flops": conv_flops,
    }

    # 3. Attention Sub-components (seq = 16 patches, d_model = 64, d_ff = 128)
    seq = 16
    d_model = 64
    d_ff = 128

    # 3a. Q, K, V projections
    qkv_params = 3 * (d_model * d_model + d_model)  # 12,480
    qkv_macs = 3 * seq * (d_model ** 2)             # 196,608
    qkv_flops = 2 * qkv_macs                        # 393,216
    costs["Attention (QKV Projections: 3x(16x64x64))"] = {
        "formula_params": "3 * (d_model^2 + d_model)",
        "params": qkv_params,
        "formula_macs": "3 * seq * d_model^2",
        "macs": qkv_macs,
        "formula_flops": "6 * seq * d_model^2",
        "flops": qkv_flops,
    }

    # 3b. Attention scores (QK^T)
    scores_macs = (seq ** 2) * d_model  # 16,384
    scores_flops = 2 * scores_macs      # 32,768
    costs["Attention (Attention Scores QK^T: 16x16x64)"] = {
        "formula_params": "0 (Dynamic matrix multiplication)",
        "params": 0,
        "formula_macs": "seq^2 * d_model",
        "macs": scores_macs,
        "formula_flops": "2 * seq^2 * d_model",
        "flops": scores_flops,
    }

    # 3c. Attention context (AV)
    context_macs = (seq ** 2) * d_model  # 16,384
    context_flops = 2 * context_macs     # 32,768
    costs["Attention (Attention Context AV: 16x16x64)"] = {
        "formula_params": "0 (Dynamic matrix multiplication)",
        "params": 0,
        "formula_macs": "seq^2 * d_model",
        "macs": context_macs,
        "formula_flops": "2 * seq^2 * d_model",
        "flops": context_flops,
    }

    # 3d. Output projection (W_O)
    out_proj_params = d_model * d_model + d_model  # 4,160
    out_proj_macs = seq * (d_model ** 2)           # 65,536
    out_proj_flops = 2 * out_proj_macs             # 131,072
    costs["Attention (Output Projection W_O: 16x64x64)"] = {
        "formula_params": "d_model^2 + d_model",
        "params": out_proj_params,
        "formula_macs": "seq * d_model^2",
        "macs": out_proj_macs,
        "formula_flops": "2 * seq * d_model^2",
        "flops": out_proj_flops,
    }

    # 3e. Total MHA Block
    mha_params = qkv_params + out_proj_params  # 16,640
    mha_macs = qkv_macs + scores_macs + context_macs + out_proj_macs  # 294,912
    mha_flops = 2 * mha_macs                                          # 589,824
    costs["Attention (MHA Block Total: seq=16, d=64)"] = {
        "formula_params": "4 * (d_model^2 + d_model)",
        "params": mha_params,
        "formula_macs": "4 * seq * d_model^2 + 2 * seq^2 * d_model",
        "macs": mha_macs,
        "formula_flops": "8 * seq * d_model^2 + 4 * seq^2 * d_model",
        "flops": mha_flops,
    }

    # 3f. Feed-Forward Network (FFN)
    ffn_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)  # 16,576
    ffn_macs = 2 * seq * d_model * d_ff                               # 262,144
    ffn_flops = 2 * ffn_macs                                          # 524,288
    costs["Attention (Feed-Forward Network FFN: 64->128->64)"] = {
        "formula_params": "2 * d_model * d_ff + d_ff + d_model",
        "params": ffn_params,
        "formula_macs": "2 * seq * d_model * d_ff",
        "macs": ffn_macs,
        "formula_flops": "4 * seq * d_model * d_ff",
        "flops": ffn_flops,
    }

    # 3g. Full Transformer Encoder Layer (MHA + FFN + LayerNorms)
    layer_params = mha_params + ffn_params + 4 * d_model  # 33,472
    layer_macs = mha_macs + ffn_macs                     # 557,056
    layer_flops = 2 * layer_macs                         # 1,114,112
    costs["Attention (Full Encoder Layer: MHA + FFN)"] = {
        "formula_params": "MHA_params + FFN_params + 4 * d_model",
        "params": layer_params,
        "formula_macs": "MHA_macs + FFN_macs",
        "macs": layer_macs,
        "formula_flops": "2 * (MHA_macs + FFN_macs)",
        "flops": layer_flops,
    }

    return costs

    return costs


def hand_calculate_full_model_costs(model_name: str, mode: str = "specified") -> Dict[str, Any]:
    """
    Calculates the exact total parameters, MACs, and FLOPs for an entire model forward pass.
    """
    model_name = model_name.upper()
    if model_name == "MLP":
        if mode == "specified":
            # 784 -> 256 -> 128 -> 10
            h1, h2 = 256, 128
        else:
            # 784 -> 144 -> 72 -> 10
            h1, h2 = 144, 72
            
        p1 = 784 * h1 + h1
        p2 = h1 * h2 + h2
        p3 = h2 * 10 + 10
        total_params = p1 + p2 + p3

        macs = 784 * h1 + h1 * h2 + h2 * 10
        flops = 2 * macs
        return {
            "model": "MLP",
            "mode": mode,
            "params": total_params,
            "macs": macs,
            "flops": flops,
        }

    elif model_name == "CNN":
        if mode == "specified":
            c1, c2 = 16, 32
        else:
            c1, c2 = 64, 128

        # Conv1: 1 -> c1, 3x3, pad 1 => out 28x28
        p_c1 = 1 * c1 * 9 + c1
        macs_c1 = 9 * 1 * c1 * 28 * 28

        # MaxPool 2x2 => 14x14
        # Conv2: c1 -> c2, 3x3, pad 1 => out 14x14
        p_c2 = c1 * c2 * 9 + c2
        macs_c2 = 9 * c1 * c2 * 14 * 14

        # MaxPool 2x2 => 7x7
        # Linear: c2 * 7 * 7 -> 10
        in_fc = c2 * 7 * 7
        p_fc = in_fc * 10 + 10
        macs_fc = in_fc * 10

        total_params = p_c1 + p_c2 + p_fc
        macs = macs_c1 + macs_c2 + macs_fc
        flops = 2 * macs
        return {
            "model": "CNN",
            "mode": mode,
            "params": total_params,
            "macs": macs,
            "flops": flops,
        }

    elif model_name in ["ATTENTION", "VIT"]:
        if mode == "specified":
            num_layers = 2
        else:
            num_layers = 4
        seq = 16
        d_model = 64
        d_ff = 128
        
        # Patch proj: Conv2d(1, 64, 7, stride=7) -> 4x4=16 patches
        p_patch = 1 * d_model * 49 + d_model
        macs_patch = 49 * 1 * d_model * 4 * 4  # 50,176
        
        # Learnable pos embed: 16 * 64 = 1,024
        p_pos = seq * d_model
        
        # Each Transformer Encoder Layer:
        # MHA:
        # QKV: 3 * seq * d_model * d_model
        # QK^T: seq * seq * d_model
        # AV: seq * seq * d_model
        # Out proj: seq * d_model * d_model
        macs_mha = 3 * seq * (d_model**2) + 2 * (seq**2) * d_model + seq * (d_model**2)
        p_mha = (d_model * 3 * d_model + 3 * d_model) + (d_model * d_model + d_model)
        
        # FFN:
        # Linear 1: seq * d_model * d_ff
        # Linear 2: seq * d_ff * d_model
        macs_ffn = 2 * seq * d_model * d_ff
        p_ffn = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
        
        # LayerNorms: 2 per layer (norm1, norm2)
        p_ln = 2 * (2 * d_model)
        
        p_layer = p_mha + p_ffn + p_ln
        macs_layer = macs_mha + macs_ffn
        
        # Head: Linear(64, 10)
        p_head = d_model * 10 + 10
        macs_head = d_model * 10
        
        total_params = p_patch + p_pos + (num_layers * p_layer) + p_head
        total_macs = macs_patch + (num_layers * macs_layer) + macs_head
        total_flops = 2 * total_macs
        return {
            "model": "Attention",
            "mode": mode,
            "params": total_params,
            "macs": total_macs,
            "flops": total_flops,
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


def profile_model(model: nn.Module, input_size: Tuple[int, ...] = (1, 1, 28, 28)) -> Dict[str, Any]:
    """
    Profiles a PyTorch model using both torchinfo and thop.
    Uses deepcopies so that profiling side-effects do not pollute the original model.
    Runs both:
      - Native (unhooked) thop: to expose unsupported operations (e.g. native Transformer blocks)
      - Verified custom-hooked thop: accurately counting all operations including TransformerEncoderLayer
    """
    import copy
    dummy_input = torch.randn(*input_size)

    # 1. torchinfo
    model_torchinfo = copy.deepcopy(model).eval().cpu()
    info = summary(model_torchinfo, input_size=input_size, verbose=0)
    torchinfo_params = info.total_params
    torchinfo_trainable = info.trainable_params
    torchinfo_mult_adds = getattr(info, "total_mult_adds", None)

    # 2. Native thop (without custom hooks)
    model_thop_raw = copy.deepcopy(model).eval().cpu()
    try:
        thop_raw_macs, thop_raw_params = thop.profile(model_thop_raw, inputs=(dummy_input,), verbose=False)
        thop_raw_macs = int(thop_raw_macs)
        thop_raw_params = int(thop_raw_params)
    except Exception:
        thop_raw_macs = None
        thop_raw_params = None

    # 3. Verified custom-hooked thop (accounting for TransformerEncoderLayer)
    model_thop_custom = copy.deepcopy(model).eval().cpu()
    try:
        thop_custom_macs, thop_custom_params = thop.profile(
            model_thop_custom,
            inputs=(dummy_input,),
            custom_ops={nn.TransformerEncoderLayer: count_transformer_encoder_layer},
            verbose=False,
        )
        thop_custom_macs = int(thop_custom_macs)
        thop_custom_params = int(thop_custom_params)
    except Exception:
        thop_custom_macs = thop_raw_macs
        thop_custom_params = thop_raw_params

    # Effective full-model compute (1 MAC = 2 FLOPs)
    effective_macs = thop_custom_macs
    effective_flops = 2 * effective_macs if effective_macs is not None else None

    return {
        "torchinfo_params": torchinfo_params,
        "torchinfo_trainable_params": torchinfo_trainable,
        "torchinfo_mult_adds": torchinfo_mult_adds,
        "thop_raw_macs": thop_raw_macs,
        "thop_raw_params": thop_raw_params,
        "thop_custom_macs": thop_custom_macs,
        "thop_custom_params": thop_custom_params,
        "effective_macs": effective_macs,
        "effective_flops": effective_flops,
        # Backward compatibility aliases
        "thop_macs": thop_custom_macs,
        "thop_params": thop_raw_params,
    }
