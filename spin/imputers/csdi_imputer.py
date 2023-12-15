from typing import Type, Mapping, Callable, Optional, Union, List

import torch
from torchmetrics import Metric
from tsl.imputers import Imputer
from tsl.predictors import Predictor
import numpy as np
import random
from torch import Tensor


import numpy as np




class CsdiImputer(Imputer):
    def __init__(self,
                 model_class: Type,
                 model_kwargs: Mapping,
                 optim_class: Type,
                 optim_kwargs: Mapping,
                 loss_fn: Callable,
                 scale_target: bool = True,
                 whiten_prob: Union[float, List[float]] = 0.2,
                 prediction_loss_weight: float = 1.0,
                 metrics: Optional[Mapping[str, Metric]] = None,
                 scheduler_class: Optional = None,
                 scheduler_kwargs: Optional[Mapping] = None,
                 num_steps=50,
                 beta_start=0.0001,
                 beta_end=0.5,
                 n_samples=10
                 ):
        super().__init__(model_class=model_class,
                                          model_kwargs=model_kwargs,
                                          optim_class=optim_class,
                                          optim_kwargs=optim_kwargs,
                                          loss_fn=loss_fn,
                                          scale_target=scale_target,
                                          whiten_prob=whiten_prob,
                                          prediction_loss_weight=prediction_loss_weight,
                                          metrics=metrics,
                                          scheduler_class=scheduler_class,
                                          scheduler_kwargs=scheduler_kwargs)
        self.num_steps = num_steps

        self.beta = np.linspace(
            beta_start ** 0.5, beta_end ** 0.5, self.num_steps
        ) ** 2


        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1).unsqueeze(1)
        self.n_samples = n_samples

    def on_train_batch_start(self, batch, batch_idx: int,
                             unused: Optional[int] = 0) -> None:

        observed_data = batch.input.x
        B, L, K, C = observed_data.shape  # [batch, steps, nodes, channels]
        device = self.device
        t = torch.randint(0, self.num_steps, [B])
        current_alpha = self.alpha_torch[t].to(device)  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        batch['noise'] = noise
        batch.input['noisy_data'] = noisy_data
        batch.input['diffusion_step'] = t.to(device)

        # randomly mask out value with probability p = whiten_prob
        batch.original_mask = mask = batch.input.mask
        p = self.whiten_prob
        if isinstance(p, Tensor):
            p_size = [mask.size(0)] + [1] * (mask.ndim - 1)
            p = p[torch.randint(len(p), p_size)].to(device=mask.device)

        whiten_mask = torch.zeros(mask.size(), device=mask.device).bool()
        time_points_observed = torch.rand(mask.size(0), mask.size(1), 1, 1, device=mask.device) > p

        # repeat along the spatial dimensions
        time_points_observed = time_points_observed.repeat(1, 1, mask.size(2), mask.size(3))

        whiten_mask[time_points_observed] = True

        batch.input.mask = mask & whiten_mask
        # whiten missing values
        if 'x' in batch.input:
            batch.input.x = batch.input.x * batch.input.mask


    def on_validation_batch_start(self, batch, batch_idx: int,
                             unused: Optional[int] = 0) -> None:
        self.on_train_batch_start(batch, batch_idx, unused)



    def shared_step(self, batch, mask):
        epsilon = batch.noise
        epsilon_hat = self.predict_batch(batch, preprocess=False,
                                                postprocess=False,
                                                return_target=False)
        loss = self.loss_fn(epsilon_hat, epsilon, mask)
        return epsilon_hat.detach(), epsilon, loss


    def training_step(self, batch, batch_idx):

        # ########################################################
        # batch.input.x = torch.zeros_like(batch.input.x)
        # batch.input.mask = torch.zeros_like(batch.input.mask)
        # ########################################################

        injected_missing = (batch.original_mask - batch.mask)
        epsilon_hat, epsilon, loss = self.shared_step(batch, mask=injected_missing)
        # epsilon_hat, epsilon, loss = self.shared_step(batch, mask=batch.original_mask)
        # Logging
        # self.train_metrics.update(epsilon_hat, epsilon, batch.original_mask)
        # self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        # self.log_loss('train', loss, batch_size=batch.batch_size)
        self.log('train_mse', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # ########################################################
        # batch.input.x = torch.zeros_like(batch.input.x)
        # batch.input.mask = torch.zeros_like(batch.input.mask)
        # ########################################################

        observed_data = batch.y

        # scale target
        if not self.scale_target:
            observed_data = batch.transform['y'].transform(observed_data)
        B, L, K, C = observed_data.shape  # [batch, steps, nodes, channels]
        device = self.device
        val_loss_sum = 0
        for t in range(self.num_steps):
            t = torch.tensor([t] * B)
            current_alpha = self.alpha_torch[t].to(device)  # (B,1,1)
            noise = torch.randn_like(observed_data)
            noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
            batch['noise'] = noise
            batch.input['noisy_data'] = noisy_data
            batch.input['diffusion_step'] = t.to(device)
            epsilon_hat, epsilon, val_loss = self.shared_step(batch, batch.eval_mask)
            val_loss_sum += val_loss.detach()

        val_loss_sum /= self.num_steps
        # Logging
        # self.val_metrics.update(epsilon_hat, epsilon, batch.eval_mask)
        # self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        # self.log_loss('val', val_loss, batch_size=batch.batch_size)
        self.log('val_mse', val_loss_sum, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss_sum

    def test_step(self, batch, batch_idx):

        # ########################################################
        # batch.input.x = torch.zeros_like(batch.input.x)
        # batch.input.mask = torch.zeros_like(batch.input.mask)
        # ########################################################

        # batch.input.target_mask = batch.eval_mask
        # Compute outputs and rescale
        observed_data = batch.input.x
        B, L, K, C = observed_data.shape  # [batch, steps, nodes, channels]
        device = self.device
        n_samples = self.n_samples
        imputed_samples = torch.zeros(n_samples, B, L, K, C).to(device)

        trans = batch.transform.get('y')

        for i in range(n_samples):
            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps-1, -1, -1):
                noisy_data = current_sample
                batch.input['noisy_data'] = noisy_data
                batch.input['diffusion_step'] = torch.tensor([t]).to(device)
                epsilon_hat = self.predict_batch(batch, preprocess=False,
                                                postprocess=False,
                                                return_target=False)
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * epsilon_hat)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                    current_sample += sigma * noise
            if trans is not None:
               current_sample = trans.inverse_transform(current_sample)
            imputed_samples[i, ...] = current_sample



        y, eval_mask = batch.y, batch.eval_mask
        y_hat = imputed_samples.median(dim=0).values





        test_loss = self.loss_fn(y_hat, y, eval_mask)
        print(test_loss)

        # Logging
        self.test_metrics.update(y_hat.detach(), y, eval_mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', test_loss, batch_size=batch.batch_size)
        return test_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):

        # ########################################################
        # batch.input.x = torch.zeros_like(batch.input.x)
        # batch.input.mask = torch.zeros_like(batch.input.mask)
        # ########################################################


        # batch.input.target_mask = batch.eval_mask
        # Compute outputs and rescale
        observed_data = batch.input.x
        B, L, K, C = observed_data.shape  # [batch, steps, nodes, channels]
        device = self.device
        n_samples = self.n_samples
        imputed_samples = torch.zeros(n_samples, B, L, K, C).to(device)

        trans = batch.transform.get('y')

        for i in range(n_samples):
            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                noisy_data = current_sample
                batch.input['noisy_data'] = noisy_data
                batch.input['diffusion_step'] = torch.tensor([t]).to(device)
                epsilon_hat = self.predict_batch(batch, preprocess=False,
                                                postprocess=False,
                                                return_target=False)
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * epsilon_hat)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                    current_sample += sigma * noise
            if trans:
                current_sample = batch.transform['y'].inverse_transform(current_sample)
            imputed_samples[i, ...] = current_sample

        y_hat = imputed_samples.median(dim=0).values
        imputed_samples = imputed_samples.permute(1, 0, 2, 3, 4)  # B, n_samples, L, K, C
        output = dict(y=batch.y, y_hat=y_hat, eval_mask=batch.eval_mask, observed_mask=batch.input.mask, imputed_samples=imputed_samples)

        if 'st_coords' in batch:
            output['st_coords'] = batch.st_coords

        return output

    @staticmethod
    def add_argparse_args(parser, **kwargs):
        parser = Predictor.add_argparse_args(parser)
        parser.add_argument('--whiten-prob', type=float, default=0.05)
        parser.add_argument('--n-samples', type=int, default=10)
        return parser