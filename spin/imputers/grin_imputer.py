from typing import Optional
import torch
from tsl.imputers import Imputer
from torch import Tensor

class GrinImputer(Imputer):

    def on_train_batch_start(self, batch, batch_idx: int,
                             unused: Optional[int] = 0) -> None:
        r"""For every training batch, randomly mask out value with probability
        :math:`p = \texttt{self.whiten\_prob}`. Then, whiten missing values in
         :obj:`batch.input.x`"""
        super(Imputer, self).on_train_batch_start(batch, batch_idx, unused)
        # randomly mask out value with probability p = whiten_prob
        batch.original_mask = mask = batch.input.mask
        p = self.whiten_prob
        if isinstance(p, Tensor):
            p_size = [mask.size(0)] + [1] * (mask.ndim - 1)
            p = p[torch.randint(len(p), p_size)].to(device=mask.device)


        whiten_mask = torch.zeros(mask.size(), device=mask.device).bool()

        batch.input.mask = mask & whiten_mask
        # whiten missing values
        if 'x' in batch.input:
            batch.input.x = batch.input.x * batch.input.mask


    def validation_step(self, batch, batch_idx):
        batch.input.x = torch.zeros_like(batch.input.x)
        super().validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        batch.input.x = torch.zeros_like(batch.input.x)
        output = super().predict_step(batch, batch_idx, dataloader_idx)
        output['eval_mask'] = output['mask']
        output['observed_mask'] = batch.input.mask
        del output['mask']

        if 'st_coords' in batch:
            output['st_coords'] = batch.st_coords

        return output
