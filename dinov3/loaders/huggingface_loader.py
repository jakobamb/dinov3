# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from typing import Dict, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download

logger = logging.getLogger("dinov3")


def _convert_hf_keys_to_dinov3(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Convert HuggingFace DINOv3 state dict keys to native DINOv3 format.

    HF DINOv3 models use different key naming conventions than the native
    implementation. This function translates between the two formats.

    Args:
        state_dict: HuggingFace model state dict

    Returns:
        Converted state dict with DINOv3 native keys
    """
    converted_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # Convert embeddings layer
        if "embeddings.patch_embeddings.projection" in new_key:
            new_key = new_key.replace(
                "embeddings.patch_embeddings.projection", "patch_embed.proj"
            )
        elif "embeddings.cls_token" in new_key:
            new_key = new_key.replace("embeddings.cls_token", "cls_token")

        # Convert encoder blocks
        if "encoder.layer." in new_key:
            new_key = new_key.replace("encoder.layer.", "blocks.")

        # Convert attention layers
        if ".attention.attention.query" in new_key:
            new_key = new_key.replace(".attention.attention.query", ".attn.qkv")
            # Note: HF splits qkv, but DINOv3 uses fused qkv - handle specially
            # Skip key weights, we'll handle in qkv fusion
        elif ".attention.output.dense" in new_key:
            new_key = new_key.replace(".attention.output.dense", ".attn.proj")

        # Convert layer norms
        if ".layernorm_before" in new_key:
            new_key = new_key.replace(".layernorm_before", ".norm1")
        elif ".layernorm_after" in new_key:
            new_key = new_key.replace(".layernorm_after", ".norm2")

        # Convert MLP layers
        if ".intermediate.dense" in new_key:
            new_key = new_key.replace(".intermediate.dense", ".mlp.fc1")
        elif ".output.dense" in new_key:
            new_key = new_key.replace(".output.dense", ".mlp.fc2")

        converted_state_dict[new_key] = value

    # Handle QKV fusion - combine separate Q, K, V weights into single qkv
    converted_state_dict = _fuse_qkv_weights(state_dict, converted_state_dict)

    return converted_state_dict


def _fuse_qkv_weights(
    original_state_dict: Dict[str, torch.Tensor],
    converted_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Fuse separate Q, K, V weights from HuggingFace format into single qkv
    weights for DINOv3.

    Args:
        original_state_dict: Original HF state dict with separate q, k, v
        converted_state_dict: Partially converted state dict

    Returns:
        State dict with fused qkv weights
    """
    # Find all transformer blocks
    block_indices = set()
    for key in original_state_dict.keys():
        if "encoder.layer." in key:
            parts = key.split(".")
            if "layer" in parts:
                layer_idx = parts[parts.index("layer") + 1]
                try:
                    block_indices.add(int(layer_idx))
                except ValueError:
                    continue

    # Fuse qkv for each block
    for block_idx in block_indices:
        # Define key patterns for q, k, v weights and biases
        base_key = f"encoder.layer.{block_idx}.attention.attention"
        q_weight_key = f"{base_key}.query.weight"
        k_weight_key = f"{base_key}.key.weight"
        v_weight_key = f"{base_key}.value.weight"

        q_bias_key = f"{base_key}.query.bias"
        k_bias_key = f"{base_key}.key.bias"
        v_bias_key = f"{base_key}.value.bias"

        # Check if all qkv weights exist
        qkv_weight_keys = [q_weight_key, k_weight_key, v_weight_key]
        if all(key in original_state_dict for key in qkv_weight_keys):
            # Fuse weights
            q_weight = original_state_dict[q_weight_key]
            k_weight = original_state_dict[k_weight_key]
            v_weight = original_state_dict[v_weight_key]
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            converted_state_dict[f"blocks.{block_idx}.attn.qkv.weight"] = qkv_weight

            # Fuse biases if they exist
            qkv_bias_keys = [q_bias_key, k_bias_key, v_bias_key]
            if all(key in original_state_dict for key in qkv_bias_keys):
                q_bias = original_state_dict[q_bias_key]
                k_bias = original_state_dict[k_bias_key]
                v_bias = original_state_dict[v_bias_key]
                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                converted_state_dict[f"blocks.{block_idx}.attn.qkv.bias"] = qkv_bias

    return converted_state_dict


def load_huggingface_model(
    model_id: str, cfg, cache_dir: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Load a DINOv3 model from HuggingFace Hub and convert it to native DINOv3
    format.

    Args:
        model_id: HuggingFace model identifier (e.g., "facebook/dinov3-vitb16")
        cfg: DINOv3 training configuration for architecture validation
        cache_dir: Optional directory to cache downloaded models

    Returns:
        State dict compatible with native DINOv3 models

    Raises:
        ValueError: If model architecture is incompatible
        RuntimeError: If model loading fails
    """
    logger.info(f"Loading HuggingFace model: {model_id}")

    try:
        # Download model files
        model_path = hf_hub_download(
            repo_id=model_id, filename="pytorch_model.bin", cache_dir=cache_dir
        )

        # Load model weights
        logger.info(f"Loading weights from {model_path}")
        state_dict = torch.load(model_path, map_location="cpu")

        # Convert keys to DINOv3 format
        logger.info("Converting HuggingFace keys to DINOv3 format")
        converted_state_dict = _convert_hf_keys_to_dinov3(state_dict)

        logger.info(f"Successfully loaded and converted HuggingFace model {model_id}")
        logger.info(f"Model contains {len(converted_state_dict)} parameters")

        return converted_state_dict

    except Exception as e:
        raise RuntimeError(
            f"Failed to load HuggingFace model {model_id}: {str(e)}"
        ) from e
