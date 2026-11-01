"""Per-process OpenSTL overrides for this isolated Hubei SimVP run only."""
from __future__ import annotations

import os
import sys
from pathlib import Path

repo = Path(os.environ.get("SIMVP_REPO", "/home/Data_Pool_3/chenyi/GHI/SimVPv2"))
if repo.is_dir() and str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

try:
    import numpy as np
    from openstl.api.exp import BaseExperiment
    from openstl.datasets import dataloader_hbagri
    from pytorch_lightning import Trainer
    import openstl.utils as openstl_utils

    # Lightning DDP re-executes the training entry point in child Python
    # processes.  Apply the same frozen config defaults in every worker so
    # argparse's generic defaults (notably mmnist/./data) cannot override it.
    config_file_s = os.environ.get("HUBEI_SIMVP_CONFIG")
    if config_file_s:
        _frozen_cfg = openstl_utils.load_config(config_file_s)
        _base_create_parser = openstl_utils.create_parser

        def _create_frozen_parser():
            parser = _base_create_parser()
            parser.set_defaults(**_frozen_cfg)
            return parser

        openstl_utils.create_parser = _create_frozen_parser

    # Keep Lightning artifacts out of the read-only source checkout and avoid
    # its incompatible default TensorBoard hyperparameter logger. The project
    # run itself already records epoch metrics and checkpoints via callbacks.
    def _init_writable_trainer(self, args, callbacks, strategy):
        run_root = Path(args.res_dir) / args.ex_name
        run_root.mkdir(parents=True, exist_ok=True)
        return Trainer(
            devices=args.gpus,
            max_epochs=args.epoch,
            strategy=strategy,
            accelerator="gpu",
            callbacks=callbacks,
            logger=False,
            default_root_dir=str(run_root),
        )

    BaseExperiment._init_trainer = _init_writable_trainer

    _base_load_callbacks = BaseExperiment._load_callbacks

    def _load_callbacks_without_logger_monitor(self, args, save_dir, ckpt_dir):
        callbacks, resolved_save_dir = _base_load_callbacks(
            self, args, save_dir, ckpt_dir
        )
        callbacks = [
            callback for callback in callbacks
            if callback.__class__.__name__ != "LearningRateMonitor"
        ]
        return callbacks, resolved_save_dir

    BaseExperiment._load_callbacks = _load_callbacks_without_logger_monitor

    norm_file_s = os.environ.get("HUBEI_SIMVP_NORM")
    if norm_file_s:
        norm_file = Path(norm_file_s)

        def _load_frozen_train_statistics(self) -> None:
            with np.load(norm_file, allow_pickle=False) as z:
                self.mean = np.asarray(z["agri_mean"], dtype=np.float32).copy()
                self.std = np.asarray(z["agri_std"], dtype=np.float32).copy()
                self.input_mean = np.asarray(z["input_mean"], dtype=np.float32).copy()
                self.input_std = np.asarray(z["input_std"], dtype=np.float32).copy()
            if self.mean.shape != (13,) or self.input_mean.shape != (16,):
                raise ValueError("Unexpected frozen Hubei normalization dimensions")

        dataloader_hbagri.HBAGRIDataset._compute_stats = _load_frozen_train_statistics

    def _closed_test(self) -> None:
        print("TEST_CLOSED: skipped by frozen Hubei train/validation contract", flush=True)

    BaseExperiment.test = _closed_test
except Exception as exc:  # leave a hard, visible marker for child-process diagnostics
    print(f"HUBEI_WORKER_HOOK_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
