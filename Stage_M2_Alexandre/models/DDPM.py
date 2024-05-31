"""
Author: Alexandre LECLERCQ
Licence: MIT
"""

import torch
import torch.nn as nn

"""
DDPM: Denoising Diffusion Probabilistic Model
"""


class DDPM(nn.Module):
    def __init__(self, model: nn.Module, timestep: int, beta_1=1e-4, beta_T=1e-2):
        super(DDPM, self).__init__()
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestep = timestep
        self.betas = torch.linspace(beta_1, beta_T, timestep)
        self.beta_bars = torch.tensor([torch.prod(self.betas[0:i + 1]) for i in range(len(self.betas))])
        self.alphas = 1 - self.betas
        self.apha_bars = torch.tensor([torch.prod(self.alphas[0:i + 1]) for i in range(len(self.alphas))])

    def q_xt_x0(self, x0, t, epsilon=None):
        """
        Generate the noise from x0 to xt
        :param x0:
        :param t:
        :return:
        """
        n, c, h, w = x0.shape  # retrieve the dimension of x0

        if epsilon is None:
            epsilon = torch.randn(n, c, h, w).to(self.device)  # sampling of a centered and reduced Gaussian distribution
        alpha_bar_t = self.apha_bars[t]

        # reshape(n, 1, 1, 1): apply the same alpha_bar for each n-dimension (batch)
        mean = alpha_bar_t.sqrt().reshape(n, 1, 1, 1) * x0
        std = (1 - alpha_bar_t).sqrt().reshape(n, 1, 1, 1)

        return mean + std * epsilon

    def training(self, x0):
        n, c, h, w = x0.shape  # retrieve the dimension of x0
        t = torch.randint(low=1, high=1001, size=(n,))  # sample uniform timestep for each image of the batch
        epsilon = torch.randn(n, c, h, w).to(self.device)
        x_t = self.q_xt_x0(x0=x0, t=t, epsilon=epsilon)
        epsilon_pred = self.model(x_t, t)

        loss(epsilon_pred, epsilon)

    def sampling(self, xT):
        n, c, h, w = xT.shape
        x = xT
        for t in range(self.timestep, 0, -1):
            z = torch.randn(n, c, h, w).to(self.device) if t > 1 else 0
            alpha_t = self.alphas[t]
            mean = x / alpha_t.sqrt() * (x - (self.betas[t] / (1 - self.apha_bars[t]).sqrt()) * self.model(x, t) )
            sigma = self.betas[t].sqrt()  # up to us to find better posibilities
            x = mean + sigma * z
        return x
