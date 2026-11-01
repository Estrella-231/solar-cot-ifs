"""Frozen Hubei 16-channel SimVP training on the node-local staged cache."""
method = "SimVP"
model_type = "gSTA"
dataname = "hbagri"

pre_seq_length = 8
aft_seq_length = 16
total_length = 24
temporal_stride = 1
in_shape = [8, 16, 256, 256]
out_channels = 13
metrics = ["mse", "mae"]

hid_S = 32
hid_T = 256
N_T = 8
N_S = 4
spatio_kernel_enc = 3
spatio_kernel_dec = 3
drop_path = 0.1

lr = 1e-3
opt = "adamw"
weight_decay = 1e-4
sched = "cosine"
min_lr = 1e-5
warmup_epoch = 5
warmup_lr = 1e-5
filter_bias_and_bn = False
early_stop_patience = 5
metric_for_bestckpt = "val_loss"
no_display_method_info = True

batch_size = 4
val_batch_size = 4
num_workers = 4
drop_last = False
gpus = [0, 1, 2, 3]
seed = 42
epoch = 50

data_root = "/tmp/zjnu_hubei_simvp_stage_20261016"
index_csv = "/tmp/zjnu_hubei_simvp_stage_20261016/index_trainval.csv"
data_dir = "/tmp/zjnu_hubei_simvp_stage_20261016/data"
aux_dir = "/tmp/zjnu_hubei_simvp_stage_20261016/aux"
