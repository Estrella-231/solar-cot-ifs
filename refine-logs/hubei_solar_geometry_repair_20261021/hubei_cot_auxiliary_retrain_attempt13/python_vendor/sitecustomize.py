"""Attempt13-only aliases for the historical OpenSTL import paths in timm 0.9.16."""
import sys
import timm.models.layers as _layers
import timm.layers.activations as _activations
import timm.models._efficientnet_blocks as _efficientnet_blocks
_layers.activations = _activations
sys.modules.setdefault("timm.models.efficientnet_blocks", _efficientnet_blocks)
