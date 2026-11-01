# Output manifest

2026-10-17：新月度R bank和CPP/序列键值门禁通过；单A800 seed42全数据P-R/P-Hybrid训练`214510`已正常完成；独立FP64复算、厚度分层诊断、英文LaTeX与中文对照稿更新并编译完成。无test payload读取。

| File | SHA256 | Role |
|---|---|---|
| EXPERIMENT_PLAN.md | f6480137d40bd20ddd3cc78a91afee7e7224670a6580477bd9e5ee9b45dbae9a | current contract |
| EXPERIMENT_TRACKER.md | EC7D5BECC1B487BACCA1653DF3078C673D294A1A750A42AE455CC2264D4BEFE5 | current execution status |
| MODEL_CONTRACT.json | 0a22231e53016432964a16f01804e68391d3f41823f2ebf879c1a909333bb4d2 | current contract |
| DATA_GATE_20261013.md | 6a0b8e516ffd4f16f07da173d792cc101dfbb55f0add55f229712662a2c4b2e0 | old-R date leakage audit; metadata only |
| old_R_split_overlap_audit.json | 5a2d1e2e74047710b0ab8934733c81387be1452c459499aa3a2c292f56c588bc | old-R/new-monthly date-key audit; no payload access |
| smoke_results/smoke_gpu0_fd201_20261013.json | 3a62c8ff85b90af6e0e99039202c28c517f320f433f89a9e34e239fa6fa49e26 | synthetic engineering smoke; no real-data claim |
| ../../scripts/build_retrieved_cot_manifests_20261013.py | a6fb17e3e69da3fbf41a47de37aed33d3ac0ff7cc1c394e719fa686de20d67bf | train/val-only metadata indexer |
| ../../scripts/smoke_retrieved_cot_forecaster_20261013.py | e0b315d5c9e87e4e7bceb32eef05a35b93390ebac7138bacd35a548bf203a2f6 | synthetic engineering smoke driver |
| manifest_preparation/manifest_audit.json | 30ae8a6bad3c8c892e0f84276c7168f9abf6f9cd148a881b58d60c5d2b5bf547 | PASS_METADATA_ONLY_TRAIN_VAL_MANIFESTS |
| manifest_preparation/unique_frames_train_val.csv | fed1326e7000f1aaf884cdfed1dfe7c48a7f10d8669b218010bd960194046a9d | unique frame timestamp keys |
| manifest_preparation/sequences_8to16_train_val.csv | 2d67d4592eff76d46c80a9373bde17923619a6405ca885bb5b468a80d03e5851 | train/val forecast windows |
| ../../scripts/retrieved_cot_forecaster_20261013.py | dd442dbdac8db1da2f83420796e3b2d98f8bf62a26506d504a5d4feb2074779a | candidate implementation |
| EXPERIMENT_PLAN.md | 99697EE7F9D8C2D71AA3D912B7B35F294A865A999BE13FE62E6B68138876F5FD | current execution scope, full-data paired screen |
| EXPERIMENT_TRACKER.md (2026-10-17 archived hash) | 44E33ECAA12CCBAB3450809F4A69AD640C66518FF1531CA0F10F9A8030C2F8F9 | prior live-status snapshot before job completion |
| results/data_gate/RESULTS_20261017.md | 5B10A0D7CD3212BF7969AF4C96B93B49972D742C0092EC129E4B4668D08B1E85 | verified mapping, sample mask and model-profile evidence |
| results/data_gate/retrieved_cot_mapping_audit_20261017.json | B873235B2CCC0944C3FECB20CF63C6F1623D39DD82B4A50237942C91140AC2E0 | exact CPP compact-index and sequence/bank key audit |
| audit_retrieved_cot_mapping_20261017.py | CFDFEBB369F683F8BA68A3EBBB4F4D56C264B6C17121307E7FC5CDA4D5171D18 | audit script |
| results/data_gate/retrieved_cot_mask_witness_20261017.json | 9344B474D133215A0631EE6C1FFD043CB808AB591FDFBE2C7E7536867AC8F31B | sampled remote train/val mask/value witness |
| witness_retrieved_cot_pixel_masks_20261017.py | A351000E04AEAE876461967A74E17C40A19DE752C64A9D30A90F0EEB417C68F5 | reproducible read-only mask/value witness |
| audit_retrieved_cot_full_bank_20261017.py | E27FFC816E2D900BF5038F7EC01ABFC6B4D3F5BA45145C40F583D8EFAB7ECD7A | 8-thread full bank audit plus frame geometry/mask pack source |
| run_full_bank_audit_20261017.pbs | 39CB2DC8C7193E5E42C5CAEBF24FDDBBEA91FCD5EEC4F9A5134395C4C963493A | PBS audit+pack launch; job 214505 queued |
| profile_retrieved_cot_forecaster_20261017.py | B1E9B60F9619DFB5E1BBF7632ADC1322A798682690250195F42D2A34FCAF1833 | actual-data GPU throughput/memory profile source |
| run_real_profile_20261017.pbs | 9E09F2DAA9115FC7564F97367EE8600C9E7C119A4EC0FB3702CF71D601B1BC1B | one-GPU profile launch; job 214503 exit 0 |
| results/profile_20261017/profile.json | 50F1BA541163FC9F7BF8FD78586F8A73291BC95423B71A458B8BF6E102EE3A8B | real-data forward/backward throughput and memory profile |
| results/data_gate/RESULTS_20261017.md |  | data provenance/mask gates and current training readiness; no accuracy claim |
| data/sequence_cpp_pack_20261017/pack_manifest.json | E7BF863F3603A55D4AEAE7C4A1FD6C15A2683F45BDCECD5F30D2D6EA431A09 | local train/val labels and sequence indices, uploaded SHA verified |
| train_retrieved_cot_forecaster_20261017.py | 4BAF9E6A3B2D1B49C563D42B69B7754C0276F6EC599E7136E5CC215096E74E0B | paired full-data P-R/P-Hybrid trainer; audit/hash gates and finite-loss check |
| run_paired_cot_fulltrain_seed42_20261017.pbs | 6FFB6FA3380DA98DE56A29191D3708E4BE909FA722252D837FAAB98CCCB7E5E7 | one-A800 full-data paired screen; job 214510 running |
| run_paired_cot_pilot_20261017.pbs | DE539A1B784D1C49FF80934D4B6BFD26831C38CBDC3CA3B26A7B2FD81D2F194D | one-GPU P-R/P-Hybrid pilot PBS entry; script now checks and binds the assigned GPU |
| profile_retrieved_cot_loader_20261017.py | DCD59D65F83A57D4AE29076B7A6E56CE967A221928A7F5A9907D17BB8AA25413 | actual end-to-end train loader/model profile |
| run_loader_profile_20261017.pbs | 107C2B95E8B34D32C34321C8D341CDE594DE3758A028F22F2A573C05BAE695DD | one-A800 loader profile PBS job, prepared but not submitted |
| retrieved_cot_forecaster_20261013_114812.py | dd442dbdac8db1da2f83420796e3b2d98f8bf62a26506d504a5d4feb2074779a | timestamped source snapshot |
| EXPERIMENT_PLAN_20261013_114812.md | c96b22d946d5e252ed967ba8ddeddf40dbf343acff06f58b44882d3aef3c3508 | timestamped plan snapshot |
| EXPERIMENT_TRACKER_20261013_114812.md | 4a57b9866c417fdf3ff11b9ee9c1797fd3fc092023fe10a397759383abe548b0 | timestamped tracker snapshot |
| EXPERIMENT_PLAN_20261013_121050.md | 806a87b46a8279cdabcfb82203c5fa98ade24b4df1472dceaa90c81f38e01b6d | timestamped plan snapshot |
| EXPERIMENT_TRACKER_20261013_121050.md | cd9c01bd2d818a3674ea6e6cd0583950a6c38ee7afdceed83c210a6c853c39d8 | timestamped tracker snapshot |
| MODEL_CONTRACT_20261013_121050.json | 0a22231e53016432964a16f01804e68391d3f41823f2ebf879c1a909333bb4d2 | timestamped model contract |
| retrieved_cot_forecaster_20261013_121050.py | dd442dbdac8db1da2f83420796e3b2d98f8bf62a26506d504a5d4feb2074779a | timestamped source snapshot |
| EXPERIMENT_PLAN_20261013_121500.md | f6480137d40bd20ddd3cc78a91afee7e7224670a6580477bd9e5ee9b45dbae9a | timestamped plan snapshot |
| EXPERIMENT_TRACKER_20261013_121500.md | cd9c01bd2d818a3674ea6e6cd0583950a6c38ee7afdceed83c210a6c853c39d8 | timestamped tracker snapshot |
| MODEL_CONTRACT_20261013_121500.json | 0a22231e53016432964a16f01804e68391d3f41823f2ebf879c1a909333bb4d2 | timestamped model contract |
| DATA_GATE_20261013_121500.md | 679c10c37fb31cea995f17bbbc7e39a378912baf3fbf68ae82636427297a1f55 | timestamped data gate audit |
| old_R_split_overlap_audit_20261013_121500.json | 5a2d1e2e74047710b0ab8934733c81387be1452c459499aa3a2c292f56c588bc | timestamped leakage audit |
| build_retrieved_cot_manifests_20261013_121250.py | a6fb17e3e69da3fbf41a47de37aed33d3ac0ff7cc1c394e719fa686de20d67bf | timestamped manifest-builder source |
| DATA_GATE_20261013_121200.md | 6a0b8e516ffd4f16f07da173d792cc101dfbb55f0add55f229712662a2c4b2e0 | timestamped data gate snapshot |

| FULLTRAIN_RESULTS_20261017.md | C98646EE91B234D532DC1FECD84D67546918F8E3FC2D31EBA955ED9162300E91 | paired full-data screen result and claim boundary |
| recompute_fulltrain_validation_fp64_20261017.py | 1265946E322A2EA785A8091090DB9391139965F3D4A3A606FEFD56AF11DB19DB | independent FP64 scorer |
| results/fp64_independent_validation_rescore_20261017.json | 8980694157CF1FB368BB018333CCDA21EA165FB00E5BE6C7153B15C74246EF5E | same-mask saved-prediction scores and source hashes |
