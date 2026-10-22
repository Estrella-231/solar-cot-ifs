# Reproduction and stored arrays

Remote CPU run: FD-201 (`4090`), `/home/Data_Pool/zjnu/hubei_upstream_domain_20261022_v1/`.

Invocation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -u run_domain_comparison.py --manifest agri_paths_trainval_staged.csv --out output_attempt1
python analyze_domain_comparison.py output_attempt1
```

Remote extraction uses Python 3.10, NumPy 2.2.6, netCDF4 1.7.4 and SciPy. Local analysis/plotting used `C:/Python314/python.exe`, NumPy 2.3.5, and Matplotlib. No environment packages were changed. The bundled document runtime lacks Matplotlib and was not used for analysis.

The input manifest is the parent `agri_paths_trainval_staged.csv`; its hash is bound in `output_attempt1/contract.json`. AGRI payload is read from the original FD archive by the same allowed BJT keys; paths in the staged manifest refer to the ZJNU copies and are not used as remote FD payload paths.

Each `output_attempt1/YYYYMMDD1000.npz` stores eight actual C13 maps, historical regional CPP COT, causal pairwise/middle displacement, future reference station neighborhoods, predictions for persistence/16/64/256, and daylight mask. These 15 compressed snapshots are preserved locally and on FD-201; they are omitted from Git to avoid duplicating satellite payloads. `ARTIFACT_SHA256.csv` inventories their hashes and all compact reports. Per-pixel results in `paired_pixels.csv` are versioned and reproduce the figures/statistics; the NPZ rescore independently agrees with CSV RMSE to <1e-10.

`source_inventory.json` records every opened CPP timestamp, coordinate tolerance, size and valid fraction, along with explicit missing sources. The source root belongs to the existing CPP producer and is read-only. No raw products, existing banks, weights or outputs were modified. This diagnostic has no checkpoint because it performs no training.

See `RESULT_REPORT.md` for the sampling scope and negative/uncertain findings. In particular, no new thick-cloud arrival event exists among the complete 16-lead sequences, so arrival-time improvement is not evaluated. The 19 fixed inits are not a full validation benchmark, and historical CPP is not interchangeable with the deployable frozen-R chain.
