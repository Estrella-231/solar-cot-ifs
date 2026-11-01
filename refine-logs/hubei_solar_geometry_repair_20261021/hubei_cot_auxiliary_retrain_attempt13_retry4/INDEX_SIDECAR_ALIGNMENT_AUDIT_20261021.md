# Attempt13 retry4: train/validation index-to-COT-sidecar alignment audit

Date: 2026-10-21 (read-only audit)

## Question

Does the integer returned by the SimVP dataset (`__getitem__` row index) select the CPP-COT sidecar row built for that same 8-history/16-future sequence?

## Audit

On `zjnu-hpc`, compared the prepared `indexes/full.csv` against `cpp_target_sidecar/{train,val}_keys.csv`. For every split-filtered row, checked sidecar `sample_index`, split name, initialization time from `aux_paths[7]`, and all 16 future target times from `aux_paths[8:24]`. Also inspected the training dataset implementation: it reads CSV records in order, filters into train/val lists without re-sorting, and returns the dataset row index; the attempt13 wrapper passes that index to the sidecar lookup. This confirms that sidecar row indexing and runtime dataset indexing share the same order.

## Result

| Split | Index rows | Sidecar rows | Key/order/time mismatches |
|---|---:|---:|---:|
| train | 4,086 | 4,086 | 0 |
| validation | 407 | 407 | 0 |

Prepared index SHA-256: `a8692c0f924b4869840e9c25d88f16c1f869fd88d41452f1db8f9879f29581f8`.

Train key table SHA-256: `9ad73c31ba0fabd14da96688f5ff9ddd14ee82fc0575754542663002a0cb632b`.

Validation key table SHA-256: `b1af4992c06a1a49a62c6a98db22153e2744c645d3e417d67ccc6100b9542775`.

The existing preflight separately verifies 19,481 matched future CPP frames against the R-pack arrays and masks in physical COT units. No test or buffer payload was read. This closes the row-index/key-alignment concern for this attempt; it does not establish that CPP is independent truth or that the auxiliary training improves forecasts.
