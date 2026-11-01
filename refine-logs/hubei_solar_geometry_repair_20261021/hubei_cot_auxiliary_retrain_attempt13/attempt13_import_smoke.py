#!/usr/bin/env python3
"""CPU-side import smoke for the exact attempt13 training stack."""
from __future__ import annotations

import sys
from pathlib import Path

# The batch environment installs the attempt-scoped API shim in python_vendor.
# Import it explicitly as well as relying on Python's sitecustomize hook.
import sitecustomize  # noqa: F401

import cv2
import fvcore
import pywt
import pytorch_lightning
import skimage
import timm
import torch


def main() -> None:
    attempt = Path(__file__).resolve().parent
    base = attempt.parent
    sys.path[:0] = [
        str(base / "simvp_code_snapshot"),
        str(attempt),
        str(base / "code_pipeline_attempt13"),
    ]

    from openstl.api.exp import BaseExperiment
    from openstl.datasets import dataloader_hbagri
    from openstl.models import SimVP_Model
    from train_cot_repaired import COTUNet
    import cot_aux_patch_attempt13

    assert BaseExperiment is not None
    assert SimVP_Model is not None
    assert COTUNet is not None
    assert cot_aux_patch_attempt13 is not None
    print(
        "ATTEMPT13_IMPORT_SMOKE_PASS "
        f"torch={torch.__version__} cv2={cv2.__version__} timm={timm.__version__} "
        f"skimage={skimage.__version__} lightning={pytorch_lightning.__version__} "
        f"pywt={pywt.__version__} dataset={dataloader_hbagri.__file__}",
        flush=True,
    )


if __name__ == "__main__":
    main()
