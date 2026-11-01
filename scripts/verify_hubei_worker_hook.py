#!/usr/bin/env python3
"""Assert the run-scoped sitecustomize hooks are active in a fresh process."""
from openstl.api.exp import BaseExperiment
from openstl.datasets import dataloader_hbagri

assert BaseExperiment.test.__name__ == "_closed_test", BaseExperiment.test.__name__
assert dataloader_hbagri.HBAGRIDataset._compute_stats.__name__ == "_load_frozen_train_statistics"
print("WORKER_HOOKS_ACTIVE", BaseExperiment.test.__name__,
      dataloader_hbagri.HBAGRIDataset._compute_stats.__name__, flush=True)
