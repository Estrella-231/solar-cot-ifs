# Manifest — COT-aware SimVP

| File | SHA256 | Role |
|---|---|---|
| EXPERIMENT_PLAN.md | 147253d8f13a977b9f512dc62b4d6b19dee1539098be50528081739a069eb6e7 | Current one-factor protocol |
| EXPERIMENT_TRACKER.md | dc2a00289e0ab0244f67f0576ad693e76e2e5a14684bd641fbf0c812edc3920d | Current implementation and queue status |
| cot_auxiliary.py | 3812cd6ad7a445a8845e99805c4e262425f8078b5866d39a86a0ca3fcaa224a2 | Differentiable frozen-R COT objective |
| train_cot_aux_simvp.py | 8fd8f5f5001e1ef05d61a32a108cf485c6402cad0d2cb73847a1a920e3809778 | Equal-budget candidate trainer and prediction banks |
| audit_cot_aux_validation.py | 834e7efb5bc29c45ab66bd9ae1d7c41b43c470e19eb1944afbc20e771e3d0fe6 | Independent FP64 paired COT audit and bootstrap gates |
| test_cot_auxiliary.py | 21faacec7e95cf114b30a1f16b4ea4149085a198e8b2683d95efc2c0d4dbb19a | Synthetic gradient/mask witness |
| test_frozen_R_real_cpu.py | 46c038ddfbd4fdf91cfcd0fd2fe9cc44aa34d9badf520049e1e70cad549579f3 | Actual frozen-R gradient witness |
| test_trainval_contract.py | 71a432729464e490751fb9285255f0c8614f5ef503114a6708efe27881cd0a9f | Train/validation split and model-contract witness |
| test_audit_cot_aux_validation.py | ea0fc3259ad12b2bec4d4d04b1ff150dc5849ccec365c4c38fd781f7aa4df23b | Synthetic paired-score/bootstrap witness |
| run_one_card_cot_aux_20261017.pbs | bc0b48da168409fae1cbdf3f86154f235452d37ac29161df2c82586c7bc342ce | One-A800 PBS profile, training, and audit launcher |
# OOF manifest and fold-0 launcher additions
| File | SHA256 | Role |
|---|---|---|
| build_future_ghi_oof_manifests.py | 3988bea9f20672d57df7dd5670a1fe090380569eb562258dc305fdfe5fc9858d | Join full official train manifest with forecast bank and create frame-disjoint date-fold manifests |
| test_build_future_ghi_oof_manifests.py | 6442a46af055ff61a74bcfead1d43b586ce9925ceb358431be93568703e90b3e | Synthetic date-fold, coverage, and frame-isolation witness |
| train_monthly_simvp_oof_20261019.py | 4b76c0b64628448fb0685ede8a28942cf5782b3f8fe7e92a5185a3f01dd8a264 | Fold trainer with fixed epoch-12 checkpoint and no heldout fold scoring |
| test_train_monthly_simvp_oof_20261019.py | 9cdd48db863588df2e7fa44f6b1f9e2ff3231e5ea60a335027ffe5d39722ab11 | Train-only dataset construction witness |
| run_future_ghi_oof_fold0.pbs | 049d78a11272ed497ac9743fd21e50a8354cbe622139ef3ffbdb0b05dfb97447 |
 One-A800 normalization, throughput profile, and fold-0 fixed-epoch run |
| predict_future_ghi_oof_fold.py | 796450561f443f19b1362bca7e524dbf13d0bc935678c1bda4c4667b7da00d16 | Fold-heldout forecast AGRI and frozen-R COT using observed-only inputs |
| test_predict_future_ghi_oof_fold.py | c5e89ca999fb63fdc4dd473c895125e2d0ab4a42ddb0405df49deb3c82417008 | Fold-heldout forecast AGRI and frozen-R COT using observed-only inputs |
| merge_future_ghi_oof_bank.py | dc078bda2a1afe046b7c7b5cb4cf36f5f504da59805629e611f5ba892fc69d6b | Merge OOF forecast arrays with the validation bank and audit key coverage |
| test_merge_future_ghi_oof_bank.py | 50a230d5680153d12d9405385b6a3bc044202a02f9948593a496028d6b3a65d3 | Merge OOF forecast arrays with the validation bank and audit key coverage |

# OOF future-COT shrinkage follow-up (prepared; not submitted)
| File | SHA256 | Role |
|---|---|---|
| train_future_ghi_heads_cot_shrink.py | 79cd28e8199f812f70b381a9a7e8edab650946f01b7e4e36586524098f0d5b57 | Four-arm AC50 head training; records whether training forecasts are five-fold OOF |
| audit_future_ghi_cot_shrink.py | 41a9f96428892504afd13d3a58c8cc17b2255ddefccdee0563c0b3047f05ef6e | Paired date-block GHI audit with OOF provenance in the result |
| test_audit_future_ghi_cot_shrink.py | 2a77b3f2bcb38f95715acfa24a6dbd4cf309ef634a10ccc6a0944e05dce56cd6 | Synthetic OOF provenance and paired-score audit witness |
| gate_future_ghi_oof_cot_shrink.py | 1bef3d7c6531db6ea28fa3af3540fdccc11ed340039c670e8981896b239c83a6 | Fail-closed check of pack, exact baseline provenance, and registered AC-AH gate |
| test_gate_future_ghi_oof_cot_shrink.py | 098be47ab0ba9e96cc747842dddc4a6de7c471dadd1cd621c15e187723f6e633 | Synthetic gate pass/fail and provenance witness |
| run_future_ghi_oof_cot_shrink.pbs | b81f45baca0bf76f34af2c029c7bc02597d67dff2cbf2d225087a35e81467ed3 | OOF-only AC50 follow-up launcher; not submitted |
# Fold-0 complete-cache recovery additions
| File | SHA256 | Role |
|---|---|---|
| verify_future_ghi_oof_cache.py | af4080dc20273a1f688937197366386f0f23b42053da4d6a2fafc1c920052654 | Verify exact train-frame cache filenames, nonempty entries, and absence of temporary writes without opening arrays |
| run_future_ghi_oof_fold0_resume_cache_reuse.pbs | 64e52289dea1e364a3f65168132159cfa0ff626551540743657d7f6f232bf0db | Resume fold 0 from its saved epoch-1 checkpoint using the verified complete training cache |

# Redundant-cache cleanup witness
| File | SHA256 | Role |
|---|---|---|
| verify_redundant_oof_cache.py | d52c7cd8ef899f4a103ecc3e502ce8caac2cd41a5c7c70492a95d83268b50e38 | Confirm the partial failed-job cache is a same-frame, same-size subset of the complete cache without opening arrays |
| cleanup_verified_redundant_fold0_cache.sh | 6a4df06357bc4af692c84c3b8f282560771f0a9aa04150dcc50c61a5cc184aec | Fail-closed cleanup of only the verified orphan partial fold-0 cache, followed by a full-cache inventory witness |

# DropPath-0 OOF multi-GPU screen (2026-10-20)
| File | SHA256 | Role |
|---|---|---|
| $143b883a27b36b2906a363efc86ffb2780d5386d96096a8a78767c71c6b0c1c14 | Fixed single-factor protocol, resource gates, and current run status |
| train_monthly_simvp_oof_ddp_20261020_rankrng.py | c5ff557f14e56b98a4c6e5328ad07a12aa0c01715b5f6e8ef41e22e9b2ed7b07 | Two-GPU SyncBatchNorm OOF trainer and complete input-to-update profile |
| run_future_ghi_oof_droppath0_fold.pbs | efe810f1ed224cfdc90c65b287860e40f23649a6c0df00584d768cf5b4787377 | Two-GPU fold runner; 12-epoch projection and memory gates before training |
| prewarm_future_ghi_oof_rawcache.py | 7a2ed20dea32c68b1f3a2c66e1d910d22a918132e4ae06ceef03d1040d52b5f8 | Train-fold-only shared node-local raw-cache prewarm with per-fold split audits |
| prewarm_future_ghi_oof_rawcache.pbs | 480937291921eae50219638a6aec28fb87e945e23ca3aa3ef3206ab930097ddc | CPU-only node23 prewarm job; test/validation/CPP data remain closed |
| EXPERIMENT_TRACKER.md | b5724a96913855d7aa837d6eb40623a95860ced488629a3443d46c02fcb8bab6 | Run-by-run outcomes, PBS state, and evidence boundary |

