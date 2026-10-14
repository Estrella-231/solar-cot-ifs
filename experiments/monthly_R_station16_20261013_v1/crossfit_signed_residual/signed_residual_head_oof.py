"""Versioned signed residual head with explicit OOF inputs for train rows."""
from __future__ import annotations

import torch
from torch import nn

from signed_residual_head import SignedResidualHead, trainable_parameter_count


class SignedResidualHeadOOF(SignedResidualHead):
    def forward(self, agri, geometry, cot, clear_sky,
                base_override=None, encoded_override=None):
        if base_override is None:
            with torch.no_grad():
                base_kt = self.features(agri, geometry)
                base_ghi = base_kt * clear_sky
                encoded = self.features.agri_encoder(torch.cat((agri, geometry), dim=1))
        else:
            if encoded_override is None:
                raise ValueError("encoded_override is required with base_override")
            if base_override.shape != clear_sky.shape:
                raise ValueError("base_override must have shape [B]")
            if encoded_override.shape != (agri.shape[0], 32, 3, 3):
                raise ValueError("encoded_override must have shape [B,32,3,3]")
            base_ghi = base_override
            encoded = encoded_override
        auxiliary_input = (cot.expand(-1, encoded.shape[1], -1, -1)
                           if self.variant == "AGRI_COT" else encoded)
        fused = torch.cat((encoded, self.aux_encoder(auxiliary_input)), dim=1)
        residual = self.residual_readout(torch.cat(
            (fused.flatten(start_dim=1), base_ghi[:, None] / 1000.0,
             clear_sky[:, None] / 1000.0), dim=1)).squeeze(-1) * 1000.0
        return torch.clamp_min(base_ghi + residual, 0.0), base_ghi, residual


__all__ = ["SignedResidualHeadOOF", "trainable_parameter_count"]
