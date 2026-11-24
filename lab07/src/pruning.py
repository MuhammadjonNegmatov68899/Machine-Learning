"""
Pruning implementations for Lab 07:
1. Global Unstructured Magnitude Pruning
2. Layer-wise Unstructured Magnitude Pruning
3. Structured Filter Pruning (L2-norm based channel removal and architectural contraction)
4. Mask retention utilities for 3-epoch fine-tuning.
"""

from typing import Dict, Any, Tuple, List, Optional
import copy
import torch
import torch.nn as nn
from lab07.src.models import BaselineCNN


def calculate_sparsity_statistics(model: nn.Module) -> Dict[str, Any]:
    """
    Computes precise parameter counts and sparsity metrics globally and per layer.
    """
    total_params = 0
    total_zeros = 0
    layer_stats = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            num = weight.numel()
            zeros = (weight == 0).sum().item()
            sparsity = zeros / num if num > 0 else 0.0

            total_params += num
            total_zeros += zeros
            layer_stats[f"{name}.weight"] = {
                "num_elements": num,
                "zeros": zeros,
                "sparsity_ratio": round(sparsity, 6),
                "sparsity_percent": round(sparsity * 100.0, 4),
            }

    overall_sparsity = total_zeros / total_params if total_params > 0 else 0.0

    return {
        "total_prunable_params": total_params,
        "total_zeros": total_zeros,
        "overall_sparsity_ratio": round(overall_sparsity, 6),
        "overall_sparsity_percent": round(overall_sparsity * 100.0, 4),
        "layers": layer_stats,
    }


def apply_global_unstructured_pruning(
    model: nn.Module, sparsity_target: float
) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
    """
    Applies global magnitude pruning across all eligible Conv2d and Linear weights.
    
    Args:
        model: PyTorch module to prune (deep-copied internally).
        sparsity_target: Desired fraction of weights to mask (e.g. 0.25 for 25%).
        
    Returns:
        pruned_model: Cloned model with masked weights set to 0.0.
        masks: Dictionary mapping parameter name to binary mask tensor {0.0, 1.0}.
    """
    pruned_model = copy.deepcopy(model)
    prunables = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prunables.append((f"{name}.weight", module.weight))

    masks: Dict[str, torch.Tensor] = {}
    if sparsity_target <= 0.0:
        for name, param in prunables:
            masks[name] = torch.ones_like(param.data)
        return pruned_model, masks

    # Collect and flatten all prunable weight tensors
    total_elements = sum(p.numel() for _, p in prunables)
    k = int(round(sparsity_target * total_elements))

    if k >= total_elements:
        # 100% sparsity safety fallback
        for name, param in prunables:
            masks[name] = torch.zeros_like(param.data)
            param.data.zero_()
        return pruned_model, masks

    flat_cat = torch.cat([p.data.abs().view(-1) for _, p in prunables])
    # Identify indices of the k smallest absolute weights
    _, smallest_indices = torch.topk(-flat_cat, k, largest=True)
    global_mask = torch.ones_like(flat_cat)
    global_mask[smallest_indices] = 0.0

    offset = 0
    for name, param in prunables:
        num = param.numel()
        layer_mask = global_mask[offset : offset + num].view_as(param.data).clone()
        masks[name] = layer_mask
        param.data.mul_(layer_mask)
        offset += num

    return pruned_model, masks


def apply_layerwise_unstructured_pruning(
    model: nn.Module, sparsity_target: float
) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
    """
    Applies magnitude pruning layer-by-layer across eligible Conv2d and Linear weights.
    
    Args:
        model: PyTorch module to prune (deep-copied internally).
        sparsity_target: Target sparsity applied uniformly to each individual layer.
        
    Returns:
        pruned_model: Cloned model with masked weights set to 0.0.
        masks: Dictionary mapping parameter name to binary mask tensor {0.0, 1.0}.
    """
    pruned_model = copy.deepcopy(model)
    masks: Dict[str, torch.Tensor] = {}

    for name, module in pruned_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            param = module.weight
            w_name = f"{name}.weight"
            num = param.numel()

            if sparsity_target <= 0.0:
                masks[w_name] = torch.ones_like(param.data)
                continue

            k = int(round(sparsity_target * num))
            if k >= num:
                masks[w_name] = torch.zeros_like(param.data)
                param.data.zero_()
                continue

            flat = param.data.abs().view(-1)
            _, smallest_indices = torch.topk(-flat, k, largest=True)
            layer_mask = torch.ones_like(flat)
            layer_mask[smallest_indices] = 0.0
            layer_mask = layer_mask.view_as(param.data).clone()

            masks[w_name] = layer_mask
            param.data.mul_(layer_mask)

    return pruned_model, masks


def apply_structured_filter_pruning(
    model: BaselineCNN, pruning_ratio: float
) -> BaselineCNN:
    """
    Applies structured filter pruning to BaselineCNN.
    Ranks filters by L2 norm and removes the smallest filters, physically contracting
    the network architecture, channel widths, and parameter count.
    
    Args:
        model: Trained BaselineCNN model.
        pruning_ratio: Fraction of filters to remove (e.g. 0.10, 0.25, 0.50).
        
    Returns:
        contracted_model: A physically smaller BaselineCNN instance with reduced parameters.
    """
    if pruning_ratio <= 0.0:
        return copy.deepcopy(model)

    device = next(model.parameters()).device

    # 1. Conv1 filter selection
    c1_total = model.c1
    k1 = max(1, round(c1_total * (1.0 - pruning_ratio)))
    l2_c1 = torch.norm(model.conv1.weight.data.view(c1_total, -1), p=2, dim=1)
    idx_c1, _ = torch.sort(torch.argsort(l2_c1, descending=True)[:k1])

    # 2. Conv2 filter selection
    c2_total = model.c2
    k2 = max(1, round(c2_total * (1.0 - pruning_ratio)))
    l2_c2 = torch.norm(model.conv2.weight.data.view(c2_total, -1), p=2, dim=1)
    idx_c2, _ = torch.sort(torch.argsort(l2_c2, descending=True)[:k2])

    # 3. Conv3 filter selection
    c3_total = model.c3
    k3 = max(1, round(c3_total * (1.0 - pruning_ratio)))
    l2_c3 = torch.norm(model.conv3.weight.data.view(c3_total, -1), p=2, dim=1)
    idx_c3, _ = torch.sort(torch.argsort(l2_c3, descending=True)[:k3])

    # 4. Conv4 filter selection
    c4_total = model.c4
    k4 = max(1, round(c4_total * (1.0 - pruning_ratio)))
    l2_c4 = torch.norm(model.conv4.weight.data.view(c4_total, -1), p=2, dim=1)
    idx_c4, _ = torch.sort(torch.argsort(l2_c4, descending=True)[:k4])

    # Instantiate physically contracted model
    pruned_model = BaselineCNN(
        in_channels=model.in_channels,
        c1=k1,
        c2=k2,
        c3=k3,
        c4=k4,
        fc_dim=model.fc_dim,
        num_classes=model.num_classes,
    ).to(device)

    # Transfer Conv1: slice output filters
    pruned_model.conv1.weight.data.copy_(model.conv1.weight.data[idx_c1, :, :, :])
    if model.conv1.bias is not None:
        pruned_model.conv1.bias.data.copy_(model.conv1.bias.data[idx_c1])

    # Transfer Conv2: slice output filters and input channels
    w_c2 = model.conv2.weight.data[idx_c2, :, :, :][:, idx_c1, :, :]
    pruned_model.conv2.weight.data.copy_(w_c2)
    if model.conv2.bias is not None:
        pruned_model.conv2.bias.data.copy_(model.conv2.bias.data[idx_c2])

    # Transfer Conv3: slice output filters and input channels
    w_c3 = model.conv3.weight.data[idx_c3, :, :, :][:, idx_c2, :, :]
    pruned_model.conv3.weight.data.copy_(w_c3)
    if model.conv3.bias is not None:
        pruned_model.conv3.bias.data.copy_(model.conv3.bias.data[idx_c3])

    # Transfer Conv4: slice output filters and input channels
    w_c4 = model.conv4.weight.data[idx_c4, :, :, :][:, idx_c3, :, :]
    pruned_model.conv4.weight.data.copy_(w_c4)
    if model.conv4.bias is not None:
        pruned_model.conv4.bias.data.copy_(model.conv4.bias.data[idx_c4])

    # Transfer FC1: input dimension contracted from c4 to k4
    pruned_model.fc1.weight.data.copy_(model.fc1.weight.data[:, idx_c4])
    if model.fc1.bias is not None:
        pruned_model.fc1.bias.data.copy_(model.fc1.bias.data)

    # Transfer FC2: untouched
    pruned_model.fc2.weight.data.copy_(model.fc2.weight.data)
    if model.fc2.bias is not None:
        pruned_model.fc2.bias.data.copy_(model.fc2.bias.data)

    return pruned_model


def apply_pruning_masks(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """
    Multiplies model weights in-place by their pruning mask to enforce zero weights.
    """
    for name, module in model.named_modules():
        w_name = f"{name}.weight"
        if w_name in masks:
            mask = masks[w_name].to(module.weight.device)
            module.weight.data.mul_(mask)


def zero_masked_gradients(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """
    Zeros out the gradients corresponding to masked (pruned) weights before optimizer step.
    """
    for name, module in model.named_modules():
        w_name = f"{name}.weight"
        if w_name in masks and module.weight.grad is not None:
            mask = masks[w_name].to(module.weight.grad.device)
            module.weight.grad.mul_(mask)


def verify_mask_adherence(
    model: nn.Module, masks: Dict[str, torch.Tensor], tolerance: float = 0.0
) -> bool:
    """
    Verifies that all weights corresponding to mask == 0 are strictly 0.0.
    """
    for name, module in model.named_modules():
        w_name = f"{name}.weight"
        if w_name in masks:
            mask = masks[w_name].to(module.weight.device)
            masked_entries = module.weight.data[mask == 0.0]
            if masked_entries.numel() > 0:
                max_abs = masked_entries.abs().max().item()
                if max_abs > tolerance:
                    return False
    return True
