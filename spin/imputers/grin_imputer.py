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

        # randomly mask out value with probability p = whiten_prob
        batch.original_mask = mask = batch.input.mask
        p = self.whiten_prob
        if isinstance(p, Tensor):
            p_size = [mask.size(0)] + [1] * (mask.ndim - 1)
            p = p[torch.randint(len(p), p_size)].to(device=mask.device)

        # ###################### missing completely for each time point #######################
        # whiten_mask = torch.zeros(mask.size(), device=mask.device).bool()
        # time_points_observed = torch.rand(mask.size(0), mask.size(1), 1, 1, device=mask.device) > p
        #
        # # repeat along the spatial dimensions
        # time_points_observed = time_points_observed.repeat(1, 1, mask.size(2), mask.size(3))
        # whiten_mask[time_points_observed] = True
        # ###################### missing completely for each time point #######################

        ####################### missing at random #######################
        # randomly set p percent of the time points to be missing
        whiten_mask = torch.rand(mask.size(), device=mask.device) < p
        ####################### missing at random #######################


        batch.input.mask = mask & whiten_mask
        # whiten missing values
        if 'x' in batch.input:
            batch.input.x = batch.input.x * batch.input.mask

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        output = super().predict_step(batch, batch_idx, dataloader_idx)
        output['eval_mask'] = output['mask']
        output['observed_mask'] = batch.input.mask
        del output['mask']

        if 'st_coords' in batch:
            output['st_coords'] = batch.st_coords

        return output
