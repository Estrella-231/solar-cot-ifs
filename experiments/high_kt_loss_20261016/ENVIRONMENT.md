# Existing local environment and exact invocation

No environment installation or dependency changes. Existing Python:
`F:/program/anaconda/envs/pytorch/python.exe`; Python3.9.23, PyTorch2.5.1,
CUDA12.1, NumPy1.26.4. Single RTX4060 Laptop 8GB. Data are the fixed remote
train/validation assets, copied under `data/` with all eight source SHA256s
verified against `source_transfer_receipt.json`.

The unchanged batch1792 actual forward/backward/AdamW profile passed with
finite loss, peak allocated0.3765GiB/reserved0.4121GiB and ~398k sample/s on
the 20-step kernel slice. This is a tiny-head kernel measurement, not complete
training throughput. Do not increase batch or alter optimizer update counts
merely to occupy memory in this controlled loss comparison. No multi-GPU.

Run from `F:/CODE_Classify/辐照度预报/solar-cot-ifs` in PowerShell:

```powershell
& 'F:/program/anaconda/envs/pytorch/python.exe' -u experiments/high_kt_loss_20261016/train_loss_pair.py --pack experiments/high_kt_loss_20261016/data/pack --r-bank experiments/high_kt_loss_20261016/data/bank --output experiments/high_kt_loss_20261016/runs/formal_v1 --batch 1792
```

Expected final sentinel: `FOUR_ARM_LOSS_PAIRED_GHI_COMPLETE`; completion file
state `COMPLETE_24_LOSS_PAIRED_RUNS`, 24 run entries, `test_payloads_read=0`.
Output must not preexist. Do not relaunch a started run or install packages.
Training, copied source caches and checkpoints remain ignored by Git;
compact predictions, profile, paired analysis and report are published.
