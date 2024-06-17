"""
Author: Alexandre LECLERCQ
Licence: MIT
"""

from typing import Optional
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
from einops import rearrange


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


"""
DDPM: Denoising Diffusion Probabilistic Model
"""
class DDPM:
    def __init__(self, denoiser: nn.Module, timestep: int, schedule_type: str):
        super(DDPM, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.denoiser = denoiser
        self.denoiser = self.denoiser.to(self.device)
        self.timestep = timestep
        self.betas = self.__get_beta_schedule(timestep, schedule_type)
        

        assert torch.all(self.betas > 0) and torch.all(self.betas <= 1)  # 'betas must be between 0 and 1'
        assert self.betas.shape == (timestep,)  # betas must be of length T

        self.alphas = 1 - self.betas
        self.reverse_alphas_sqrt = 1 / self.alphas.sqrt()
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_sqrt = self.alpha_bars.sqrt()
        self.one_minus_alpha_bars = 1 - self.alpha_bars
        self.one_minus_alpha_bars_sqrt = self.one_minus_alpha_bars.sqrt()
        alpha_bars_prev = torch.tensor([1, *self.alpha_bars[:-1]])  # alpha_bar_{t-1} used to compute beta_tilde
        self.beta_tilde = (1 - alpha_bars_prev) / (1 - self.alpha_bars) * self.betas 
        self.sigmas = self.betas.sqrt()

    @staticmethod
    def __get_beta_schedule(timestep: int, schedule_type: str) -> torch.Tensor:
        """
        :param timestep:
        :return: betas vector of shape : [T]
        """
        if schedule_type == 'cosine':  # Nichol and Dhariwal proposition
            s = 8e-3
            t_steps = torch.linspace(0, timestep, timestep + 1)
            alpha_bars = torch.cos( (t_steps / timestep + s) / (1 + s) * 0.5 * torch.pi) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]
            betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
            return torch.clip(betas, 0.0001, 0.9999)  # ensure that beta's values are always between ]0, 1[
        elif schedule_type == 'linear':  # Ho, et al.
            beta_1 = 1e-4
            beta_T = 2e-2
            return torch.linspace(beta_1, beta_T, timestep)
        else:
            raise NotImplementedError(schedule_type)

    @staticmethod
    def __reshape_parameters(parameters: torch.Tensor) -> torch.Tensor:
        return rearrange(parameters, 'b -> b 1 1 1')

    def forward_sampling(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Generate the noise from x0 to xt (forward process)
        :param x0: noiseless data: batch x channel x height x width
        :param t: batch of timestep: batch x 1
        :param noise:
        :return: batch of xt: batch x channel x height x width
        """
        if noise is None:
            noise = torch.randn_like(x0, device=self.device)  # sampling of a centered and reduced Gaussian distribution

        alpha_bars_sqrt_t = extract(self.alpha_bars_sqrt, t, x0.shape)
        std = extract(self.one_minus_alpha_bars_sqrt, t, x0.shape)
        mean = alpha_bars_sqrt_t * x0
        
        return mean + std * noise

    @torch.no_grad()
    def one_step_reverse_sampling(self, xt: torch.Tensor, t: torch.Tensor):
        """
        performe one reverse process from x_t to x_{t-1}
        :param xt: batch x channel x height x width
        :param t: [batch], the value is the same everywhere are we are always having batch of images at the same step
        :return: batch x channel x height x width
        """
        reverse_alphas_sqrt_t = extract(self.reverse_alphas_sqrt, t, xt.shape)
        betas_t = extract(self.betas, t, xt.shape)
        one_minus_alpha_bars_sqrt_t = extract(self.one_minus_alpha_bars_sqrt, t, xt.shape)

        predicted_noise = self.denoiser(xt, t)
        print(f't: {t}')
        print(f'epsilon_theta: {predicted_noise}\n')
        mean = reverse_alphas_sqrt_t * (xt - betas_t * predicted_noise / one_minus_alpha_bars_sqrt_t)

        if t[0] == 0:
            return mean
        else:
            z = torch.randn_like(xt, device=self.device)
            #sigma = extract(self.sigmas, t, xt.shape)
            sigma = extract(self.beta_tilde, t, xt.shape).sqrt()
            return mean + sigma * z

    @torch.no_grad()
    def rec_reverse_sampling(self, img_batch: torch.Tensor):
        """
        reconstruct images from noisy images
        :param img_batch: batch x channel x height x width
        :return: List[Tensor: batch x channel x height x width]: T x 1
        """
        batch_size = img_batch.shape[0]
        batch_image_reverse_step = [img_batch]

        for t in tqdm(reversed(range(self.timestep)), desc="reverse process"):
            t_batch = torch.full((batch_size,), t, dtype=torch.int64, device=self.device)
            img_batch = self.one_step_reverse_sampling(batch_image_reverse_step[-1], t_batch)
            batch_image_reverse_step.append(img_batch)
        return batch_image_reverse_step

    @torch.no_grad()
    def generate_reverse_sampling(self, img_shape: torch.Size):
        """
        generate an images from a batch of sampling gaussian noises
        :param img_batch: batch x channel x height x width
        :return: List[Tensor: batch x channel x height x width]: T x 1
        """
        img_batch = torch.randn(img_shape, device=self.device)

        return self.rec_reverse_sampling(img_batch)

    def compute_loss(self, x0_batch, noise=None):
        batch_size = x0_batch.shape[0]

        # sampling a timestep for each image of the batch
        t = torch.randint(low=0, high=self.timestep, size=(batch_size,), device=self.device)
        if noise is None:
            noise = torch.randn_like(x0_batch)

        x_batch_noisy = self.forward_sampling(x0=x0_batch, t=t, noise=noise)  # forward diffusion process

        predicted_noise = self.denoiser(x_batch_noisy, t)

        loss_simple = nn.functional.mse_loss(noise, predicted_noise)

        return loss_simple
