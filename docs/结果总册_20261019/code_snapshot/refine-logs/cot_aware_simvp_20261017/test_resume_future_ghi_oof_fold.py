"""Small metadata-only guard for resuming an OOF epoch checkpoint."""
from __future__ import annotations

from resume_future_ghi_oof_fold import validate_resume


def main():
    metadata = {
        "fold": 0, "seed": 42, "batch": 20, "train_samples": 25839,
        "manifest_sha256": "manifest", "normalization_sha256": "norm",
        "model_code_sha256": "model", "data_code_sha256": "data",
        "simvp_code_sha256": "simvp", "initialization": "random",
        "heldout_payload_read": False, "test_payloads_read": 0,
    }
    checkpoint = {"metadata": dict(metadata), "epoch": 1, "step": 1292}
    expected = {**metadata, "checkpoint_epoch": 1, "checkpoint_step": 1292}
    validate_resume(checkpoint, metadata, expected)
    bad = dict(expected, fold=1)
    try:
        validate_resume(checkpoint, metadata, bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("must refuse a checkpoint from another fold")
    print("OOF_RESUME_METADATA_WITNESS_PASS")


if __name__ == "__main__":
    main()
