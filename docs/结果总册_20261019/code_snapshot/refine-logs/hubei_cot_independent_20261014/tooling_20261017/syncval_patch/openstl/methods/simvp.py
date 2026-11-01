import torch
import os.path as osp
from openstl.models import SimVP_Model
from openstl.utils import check_dir
from .base_method import Base_method


class SimVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, **args):
        super().__init__(**args)

    def _build_model(self, **args):
        return SimVP_Model(**args)

    def forward(self, batch_x, batch_y=None, batch_aux=None, **kwargs):
        pre_seq_length, aft_seq_length = self.hparams.pre_seq_length, self.hparams.aft_seq_length
        if aft_seq_length == pre_seq_length:
            pred_y = self.model(batch_x)
        elif aft_seq_length < pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :aft_seq_length]
        elif aft_seq_length > pre_seq_length:
            pred_y = []
            d = aft_seq_length // pre_seq_length
            m = aft_seq_length % pre_seq_length
            
            cur_seq = batch_x.clone()
            for step_idx in range(d):
                cur_pred = self.model(cur_seq)
                pred_y.append(cur_pred)
                if batch_aux is not None and (step_idx < d - 1 or m != 0):
                    aux_start = pre_seq_length + step_idx * pre_seq_length
                    aux_end = aux_start + pre_seq_length
                    cur_aux = batch_aux[:, aux_start:aux_end]
                    cur_seq = torch.cat([cur_pred, cur_aux], dim=2)
                else:
                    cur_seq = cur_pred

            if m != 0:
                cur_pred = self.model(cur_seq)
                pred_y.append(cur_pred[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y
    
    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            batch_x, batch_y, batch_aux = batch
        else:
            batch_x, batch_y = batch
            batch_aux = None
        pred_y = self(batch_x, batch_y, batch_aux=batch_aux)
        loss = self.criterion(pred_y, batch_y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            batch_x, batch_y, batch_aux = batch
        else:
            batch_x, batch_y = batch
            batch_aux = None
        pred_y = self(batch_x, batch_y, batch_aux=batch_aux)
        loss = self.criterion(pred_y, batch_y)
        # Synchronize validation metrics across DDP ranks before ModelCheckpoint
        # and EarlyStopping inspect callback_metrics. With DistributedSampler,
        # the padded validation sampler can repeat at most one row; the final
        # selected checkpoint is still independently rescored on all 407 rows.
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        val_mae = torch.nn.functional.l1_loss(pred_y, batch_y)
        self.log("val_mae", val_mae, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        if len(batch) == 3:
            batch_x, batch_y, batch_aux = batch
        else:
            batch_x, batch_y = batch
            batch_aux = None
        pred_y = self(batch_x, batch_y, batch_aux=batch_aux)
        return {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}

    def test_epoch_end(self, outputs):
        results_all = {}
        for k in outputs[0].keys():
            import numpy as np
            results_all[k] = np.concatenate([batch[k] for batch in outputs], axis=0)

        if self.trainer.is_global_zero:
            folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))
            for np_data in ['inputs', 'trues', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])

            if self.hparams.dataname == 'hbagri':
                from openstl.utils.hbagri_eval import save_hbagri_eval_outputs
                eval_dir = check_dir(osp.join(self.hparams.save_dir, 'eval_2h4h'))
                prefix = getattr(self, 'eval_prefix', 'test')
                skill = save_hbagri_eval_outputs(
                    results_all['inputs'], results_all['preds'], results_all['trues'],
                    self.hparams.test_mean, self.hparams.test_std,
                    eval_dir, prefix=prefix, max_samples=6)
                results_all['metrics'] = np.array([
                    skill['overall_mae'],
                    skill['overall_rmse'],
                    skill['overall_corr'],
                ])
                np.save(osp.join(folder_path, 'metrics.npy'), results_all['metrics'])
            else:
                from openstl.core import metric
                eval_res, eval_log = metric(
                    results_all['preds'], results_all['trues'],
                    self.hparams.test_mean, self.hparams.test_std,
                    metrics=self.metric_list,
                    channel_names=self.channel_names,
                    spatial_norm=self.spatial_norm)
                from openstl.utils import print_log
                print_log(eval_log)
                results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])
                np.save(osp.join(folder_path, 'metrics.npy'), results_all['metrics'])
        return results_all
