#!/usr/bin/env python3
"""Launch official OpenSTL SimVP with the frozen Hubei train-only norm/index."""
from __future__ import annotations

import os
import json,hashlib,shutil
import runpy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def main() -> None:
    repo = Path(os.environ.get("SIMVP_REPO", "/home/Data_Pool_3/chenyi/GHI/SimVPv2"))
    norm_file = Path(os.environ["HUBEI_SIMVP_NORM"])
    ready=json.loads(Path(os.environ["HUBEI_SOLAR_DATA_READY"]).read_text())
    assert ready["state"]=="CORRECTED_SOLAR_TRAIN_VAL_DATA_READY"
    config_file = Path(os.environ["HUBEI_SIMVP_CONFIG"])
    if not (repo / "tools/train.py").is_file():
        raise FileNotFoundError(repo / "tools/train.py")
    if not norm_file.is_file() or not config_file.is_file():
        raise FileNotFoundError("Missing frozen normalization or config file")
    sys.path.insert(0, str(repo))
    code_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(code_dir))
    sys.path.insert(0, str(code_dir.parent / "code_pipeline_attempt13"))

    from openstl.api.exp import BaseExperiment
    from openstl.datasets import dataloader_hbagri
    from openstl.methods.simvp import SimVP
    import openstl.utils as openstl_utils
    import cot_aux_patch_attempt13 as cotpatch

    cotpatch.patch_dataset_index(dataloader_hbagri.HBAGRIDataset)
    cotpatch.patch_simvp_class(SimVP)

    # This OpenSTL checkout parses CLI arguments into `args`, merges the config
    # into a separate dict, and then mistakenly constructs BaseExperiment(args)
    # without applying that dict.  Make the frozen config the parser defaults;
    # explicit CLI options (GPU IDs, run name, etc.) still take precedence.
    cfg = openstl_utils.load_config(str(config_file))
    base_create_parser = openstl_utils.create_parser

    def create_frozen_parser():
        parser = base_create_parser()
        parser.set_defaults(**cfg)
        return parser

    openstl_utils.create_parser = create_frozen_parser

    experiment_holder = {}
    base_experiment_init = BaseExperiment.__init__

    def checked_experiment_init(self, args, dataloaders=None, strategy="ddp"):
        expected = {
            "dataname": "hbagri",
            "pre_seq_length": 8,
            "aft_seq_length": 16,
            "total_length": 24,
            "data_root": str(Path(cfg["data_root"])),
            "index_csv": str(Path(cfg["index_csv"])),
            "epoch": 50,
            "batch_size": int(cfg["batch_size"]),
            "model_type": "gSTA",
            "metric_for_bestckpt": "val_cot_rmse",
            "seed": 42,
        }
        mismatches = {
            key: (getattr(args, key, None), value)
            for key, value in expected.items()
            if getattr(args, key, None) != value
        }
        if mismatches:
            raise RuntimeError(f"Frozen Hubei SimVP config mismatch: {mismatches}")
        if int(getattr(args, "early_stop_patience", 0)) < int(cfg["epoch"]):
            raise RuntimeError("Attempt13 requires fixed equal training updates; early stopping must be disabled")
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if bool(getattr(args, "dist", False)) and world_size > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if not dist.is_initialized():
                torch.cuda.set_device(local_rank)
                dist.init_process_group(backend="nccl", init_method="env://")
            print(
                "HUBEI_SIMVP_DDP_INITIALIZED "
                f"rank={dist.get_rank()} world_size={dist.get_world_size()} local_rank={local_rank}",
                flush=True,
            )
        print(
            "HUBEI_SIMVP_EFFECTIVE_CONFIG "
            f"dataname={args.dataname} data_root={args.data_root} "
            f"index_csv={args.index_csv} input={args.pre_seq_length} "
            f"output={args.aft_seq_length} epoch={args.epoch} "
            f"batch={args.batch_size} model={args.model_type} gpus={args.gpus}",
            flush=True,
        )
        base_experiment_init(self, args, dataloaders=dataloaders, strategy=strategy)
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("Attempt13 paired experiment is restricted to one allocated GPU")
        from train_cot_repaired import COTUNet
        r_ckpt = Path(os.environ["HUBEI_COT_R_CHECKPOINT"])
        r_norm = Path(os.environ["HUBEI_COT_R_NORM"])
        sidecar = Path(os.environ["HUBEI_COT_TARGET_SIDECAR"])
        lambda_path = Path(os.environ["HUBEI_COT_LAMBDA_RECEIPT"])
        payload = torch.load(r_ckpt, map_location="cpu")
        if "model" not in payload:
            raise ValueError("Frozen corrected R checkpoint lacks its model state")
        r_model = COTUNet()
        r_model.load_state_dict(payload["model"], strict=True)
        self.method.to(torch.device("cuda"))
        init_receipt = cotpatch.attach_state(
            self.method, r_model=r_model, s_norm_path=norm_file,
            r_norm_path=r_norm, sidecar_dir=sidecar,
            mode=os.environ["HUBEI_COT_AUX_MODE"],
            lambda_path=lambda_path, r_checkpoint=r_ckpt,
        )
        lambda_receipt = cotpatch.initialize_lambda_or_load(
            self.method, self.data.train_loader.dataset, lambda_path
        )
        init_receipt["lambda_receipt_sha256"] = hashlib.sha256(lambda_path.read_bytes()).hexdigest()
        init_receipt["lambda"] = float(lambda_receipt["lambda"])
        init_path = Path(os.environ["HUBEI_COT_RUN_DIR"]) / "attempt13_init_receipt.json"
        init_path.parent.mkdir(parents=True, exist_ok=True)
        init_path.write_text(json.dumps(init_receipt, indent=2), encoding="utf-8")
        print(
            "ATTEMPT13_AUX_INITIALIZED "
            f"arm={os.environ['HUBEI_COT_AUX_MODE']} "
            f"initial_sha256={init_receipt['initial_simvp_state_sha256']} "
            f"lambda={init_receipt['lambda']:.12g} "
            f"lambda_source={lambda_receipt['split_used_for_calibration']}",
            flush=True,
        )
        experiment_holder["experiment"] = self

    BaseExperiment.__init__ = checked_experiment_init

    def load_frozen_train_statistics(self) -> None:
        with np.load(norm_file, allow_pickle=False) as z:
            self.mean = np.asarray(z["agri_mean"], dtype=np.float32).copy()
            self.std = np.asarray(z["agri_std"], dtype=np.float32).copy()
            self.input_mean = np.asarray(z["input_mean"], dtype=np.float32).copy()
            self.input_std = np.asarray(z["input_std"], dtype=np.float32).copy()
        if self.mean.shape != (13,) or self.std.shape != (13,):
            raise ValueError("Frozen AGRI normalization must have 13 channels")
        if self.input_mean.shape != (16,) or self.input_std.shape != (16,):
            raise ValueError("Frozen input normalization must have 16 channels")
        if not (np.isfinite(self.input_mean).all() and np.isfinite(self.input_std).all()
                and (self.input_std > 0).all()):
            raise ValueError("Invalid frozen training normalization")
        self.norm_sha256 = __import__("hashlib").sha256(norm_file.read_bytes()).hexdigest()
        print(f"FROZEN_TRAIN_NORM {norm_file} sha256={self.norm_sha256}", flush=True)

    dataloader_hbagri.HBAGRIDataset._compute_stats = load_frozen_train_statistics

    # The test set remains closed. tools/train.py calls exp.test() at the end
    # unless a flag exists in that upstream version; replace only that terminal
    # hook so no test/buffer sample can be loaded or scored.
    def closed_test(self) -> None:
        print("TEST_CLOSED: skipped by frozen Hubei train/validation contract", flush=True)

    BaseExperiment.test = closed_test
    # Lightning's default CSVLogger writes under the current working directory.
    # The upstream checkout is not writable by the experiment account, so keep
    # all implicit logger artifacts inside this run's own writable directory.
    run_cwd = Path(os.environ["HUBEI_SIMVP_RUN_CWD"])
    run_cwd.mkdir(parents=True, exist_ok=True)
    os.chdir(run_cwd)
    sys.argv = [str(repo / "tools/train.py"), *sys.argv[1:], "--config_file", str(config_file)]
    try:
        runpy.run_path(str(repo / "tools/train.py"), run_name="__main__")
        if int(os.environ.get('RANK','0')) != 0: return
        exp = experiment_holder['experiment']; cb = exp.trainer.checkpoint_callback
        best = Path(cb.best_model_path); assert best.is_file()
        # Read the callback's actual best at completion, not a possibly stale copied alias.
        alias = best.parent/'best.ckpt'
        if best != alias: shutil.copyfile(best,alias)
        out = Path(os.environ['HUBEI_SOLAR_RUN_DIR'])
        role=os.environ['HUBEI_SOLAR_ROLE'];meta={'state':'ATTEMPT13_SIMVP_COT_AUX_ARM_COMPLETE','role':role,'arm':os.environ['HUBEI_COT_AUX_MODE'],'random_initialization':True,'best_checkpoint':str(alias),'best_checkpoint_sha256':hashlib.sha256(alias.read_bytes()).hexdigest(),'callback_best_source':str(best),'callback_best_score':float(cb.best_model_score),'selection_metric':'equal-lead validation CPP-common-support physical COT RMSE','global_updates':int(exp.trainer.global_step),'completed_epochs':int(exp.trainer.current_epoch),'configured_max_epochs':int(cfg['epoch']),'micro_batch':int(cfg['batch_size']),'world_size':int(os.environ.get('WORLD_SIZE','1')),'input_normalization_sha256':hashlib.sha256(norm_file.read_bytes()).hexdigest(),'index_sha256':hashlib.sha256(Path(cfg['index_csv']).read_bytes()).hexdigest(),'config_sha256':hashlib.sha256(config_file.read_bytes()).hexdigest(),'solar_geometry_corrected':True,'test_payloads_read':0,'buffer_payloads_read':0}
        init_receipt=json.loads((out/'attempt13_init_receipt.json').read_text(encoding='utf-8'));meta.update(init_receipt)
        meta['lambda_receipt']=os.environ['HUBEI_COT_LAMBDA_RECEIPT']
        meta['cot_target_sidecar']=os.environ['HUBEI_COT_TARGET_SIDECAR']
        meta['frozen_r_checkpoint']=os.environ['HUBEI_COT_R_CHECKPOINT']
        meta['leadwise_val_cot_rmse_physical'] = exp.method.__dict__.get('_attempt13_last_val_lead_rmse')
        (out/'run_metadata.json').write_text(json.dumps(meta,indent=2))
        print('ATTEMPT13_SIMVP_COT_AUX_ARM_COMPLETE '+role+' '+os.environ['HUBEI_COT_AUX_MODE'],flush=True)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
