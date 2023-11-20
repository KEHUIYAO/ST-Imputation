from typing import Type, Mapping, Callable, Optional, Union, List

import torch
from torchmetrics import Metric
from tsl.imputers import Imputer
from tsl.predictors import Predictor


class InterpolationImputer(Imputer):
    def training_step(self, batch, batch_idx):
        injected_missing = (batch.original_mask - batch.mask)
        if 'target_nodes' in batch:
            injected_missing = injected_missing[..., batch.target_nodes, :]
        # batch.input.target_mask = injected_missing
        y_hat, y, loss = self.shared_step(batch, mask=injected_missing)

        # Logging
        self.train_metrics.update(y_hat, y, batch.eval_mask)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss, batch_size=batch.batch_size)
        if 'target_nodes' in batch:
            torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        # batch.input.target_mask = batch.eval_mask
        y_hat, y, val_loss = self.shared_step(batch, batch.eval_mask)

        # Logging
        self.val_metrics.update(y_hat, y, batch.eval_mask)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', val_loss, batch_size=batch.batch_size)
        return val_loss

    def test_step(self, batch, batch_idx):
        # batch.input.target_mask = batch.eval_mask
        # Compute outputs and rescale
        y_hat = self.predict_batch(batch, preprocess=False, postprocess=True)

        if isinstance(y_hat, (list, tuple)):
            y_hat = y_hat[0]

        y, eval_mask = batch.y, batch.eval_mask
        test_loss = self.loss_fn(y_hat, y, eval_mask)

        # Logging
        self.test_metrics.update(y_hat.detach(), y, eval_mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', test_loss, batch_size=batch.batch_size)
        return test_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        output = super().predict_step(batch, batch_idx, dataloader_idx)
        output['eval_mask'] = output['mask']
        output['observed_mask'] = batch.input.mask
        del output['mask']
        if 'st_coords' in batch:
            output['st_coords'] = batch.st_coords
        return output