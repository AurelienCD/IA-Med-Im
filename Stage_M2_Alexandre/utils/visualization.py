import matplotlib.pyplot as plt
import torch
from models.DDPM import DDPM


def plot_graph(x, y, xlabel='', ylabel='', title='') -> None:
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_reverse_diffusion(sec_image: torch.Tensor, sequence_t: list, diffusion_step: int):
    c, h, w = sec_image[0][0].shape
    num_cols = len(sequence_t)
    fig, axs = plt.subplots(figsize=(num_cols * 2, 2), nrows=1, ncols=num_cols, squeeze=False)
    plt.axis('off')
    for col_idx, t_val in enumerate(sequence_t):
        axs[0, col_idx].axis('off')
        axs[0, col_idx].set_title(f't: {diffusion_step-t_val}')
        axs[0, col_idx].imshow(sec_image[t_val][0].cpu().reshape(h, w, c), cmap='gray')


def plot_forward_diffusion(diffusion_model: DDPM, x_start: torch.Tensor, sequence_t: list) -> torch.Tensor:
    """

    :param diffusion_model:
    :param x_start: The image that we are diffusing inside a batch of one : 1 x C x H x W
    :param sequence_t: list of sequence to display E.g [0, 20, 60, 150, 299]
    :return: the noisy image (last step of diffusion process): batch x C x H x W
    """
    c, h, w = x_start.shape
    device = x_start.device
    num_cols = len(sequence_t)
    fig, axs = plt.subplots(figsize=(num_cols * 2, 2), nrows=1, ncols=num_cols, squeeze=False)
    plt.axis('off')
    x_noisy = None
    for col_idx, t_val in enumerate(sequence_t):
        t = torch.tensor([t_val], device=device)
        x_noisy = diffusion_model.forward_sampling(x0=x_start[None, :], t=t)
        axs[0, col_idx].axis('off')
        axs[0, col_idx].set_title(f't: {t_val}')
        axs[0, col_idx].imshow(x_noisy[0].cpu().reshape(h, w, c), cmap='gray')
    return x_noisy
