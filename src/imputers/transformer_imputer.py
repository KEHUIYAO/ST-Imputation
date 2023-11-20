from typing import Type, Mapping, Callable, Optional, Union, List

import numpy as np
import torch
from torchmetrics import Metric
from tsl.imputers import Imputer
from tsl.predictors import Predictor
from torch import Tensor



class TransformerImputer(Imputer):

    def __init__(self,
                 model_class: Type,
                 model_kwargs: Mapping,
                 optim_class: Type,
                 optim_kwargs: Mapping,
                 loss_fn: Callable,
                 scale_target: bool = True,
                 whiten_prob: Union[float, List[float]] = 0.2,
                 metrics: Optional[Mapping[str, Metric]] = None,
                 scheduler_class: Optional = None,
                 scheduler_kwargs: Optional[Mapping] = None):
        super().__init__(model_class=model_class,
                                          model_kwargs=model_kwargs,
                                          optim_class=optim_class,
                                          optim_kwargs=optim_kwargs,
                                          loss_fn=loss_fn,
                                          scale_target=scale_target,
                                          whiten_prob=whiten_prob,
                                          metrics=metrics,
                                          scheduler_class=scheduler_class,
                                          scheduler_kwargs=scheduler_kwargs)



    def on_train_batch_start(self, batch, batch_idx: int,
                             unused: Optional[int] = 0) -> None:
        r"""For every training batch, randomly mask out value with probability
        :math:`p = \texttt{self.whiten\_prob}`. Then, whiten missing values in
         :obj:`batch.input.x`"""
        # randomly mask out value with probability p = whiten_prob
        batch.original_mask = mask = batch.input.mask
        p = self.whiten_prob
        if isinstance(p, Tensor):
            p_size = [mask.size(0)] + [1] * (mask.ndim - 1)
            p = p[torch.randint(len(p), p_size)].to(device=mask.device)

        ###################### missing completely for each time point #######################
        whiten_mask = torch.zeros(mask.size(), device=mask.device).bool()
        time_points_observed = torch.rand(mask.size(0), mask.size(1), 1, 1, device=mask.device) > p

        # repeat along the spatial dimensions
        time_points_observed = time_points_observed.repeat(1, 1, mask.size(2), mask.size(3))
        whiten_mask[time_points_observed] = True
        ###################### missing completely for each time point #######################

        # ####################### missing at random #######################
        # # randomly set p percent of the time points to be missing
        # whiten_mask = torch.rand(mask.size(), device=mask.device) < p
        # ####################### missing at random #######################



        batch.input.mask = mask & whiten_mask
        # whiten missing values
        if 'x' in batch.input:
            batch.input.x = batch.input.x * batch.input.mask

    def training_step(self, batch, batch_idx):
        injected_missing = (batch.original_mask - batch.mask)
        if 'target_nodes' in batch:
            injected_missing = injected_missing[..., batch.target_nodes, :]
        # batch.input.target_mask = injected_missing
        y_hat, y, loss = self.shared_step(batch, mask=injected_missing)

        # Logging
        #self.train_metrics.update(y_hat, y, batch.eval_mask)
        self.train_metrics.update(y_hat, y, injected_missing)
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

    @staticmethod
    def add_argparse_args(parser, **kwargs):
        parser = Predictor.add_argparse_args(parser)
        parser.add_argument('--whiten-prob', type=float, default=0.05)
        return parser
