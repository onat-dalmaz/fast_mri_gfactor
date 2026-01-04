from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Union

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sigpy as sp

from mr_recon.block import Block
from mr_recon.dtypes import complex_dtype, np_complex_dtype


__all__ = [
    'LLRHparams', 'LocallyLowRank',
]

"""
A proximal gradient is defined as 
prox_g(w) = argmin_x 1/2 ||x - w||^2 + g(x)
"""

@dataclass
class LLRHparams:
    block_size: Union[Tuple[int, int],
                      Tuple[int, int, int]]
    block_stride: Union[Tuple[int, int],
                        Tuple[int, int, int]]
    threshold: float
    rnd_shift: int = 3

def soft_thresh(x: torch.Tensor,
                rho: float) -> torch.Tensor:
    """
    Soft thresholding operator

    Parameters:
    -----------
    x : torch.Tensor
        input tensor
    rho : float
        threshold value

    Returns:
    --------
    x_thresh : torch.Tensor
        thresholded tensor
    """
    return torch.exp(1j * torch.angle(x)) * torch.max(torch.abs(x) - rho, torch.zeros(x.shape, device=x.device))

class L1Wav(nn.Module):
    """Wavelet proximal operator mimicking Sid's implimentation"""

    def __init__(self, 
                 shape: tuple, 
                 lamda: float, 
                 axes: Optional[tuple] = None,
                 rnd_shift: Optional[int] = 3,
                 wave_name: Optional[str] = 'db4'):
        """
        Parameters:
        -----------
        shape - tuple
            the image/volume dimensions
        lamda - float
            Regularization strength
        axes - tuple
            axes to compute wavelet transform over
        rnd_shift - int
            randomly shifts image by rnd_shift in each dim before applying prox
        wave_name - str
            the type of wavelet to use from:
            ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus',
            'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor', ...]
            see https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#wavelet-families
        """
        super().__init__()

        # Using a fixed random number generator so that recons are consistent
        self.rng = np.random.default_rng(1000)
        self.rnd_shift = rnd_shift

        # Save wavelet params
        if axes is None:
            axes = tuple([i for i in range(len(shape))])
        self.lamda = lamda
        self.axes = axes
        self.W = sp.linop.Wavelet(shape, axes=axes, wave_name=wave_name)

    def forward(self, 
                input: torch.tensor,
                alpha: Optional[float] = 1.0):
        """
        Proximal operator for l1 wavelet

        Parameters
        ----------
        input - torch.tensor <complex> | CPU
            image/volume input
        alpha - float
            proximal 'alpha' term
        """
        
        # Random stuff
        shift = round(self.rng.uniform(-self.rnd_shift, self.rnd_shift))
        phase = np.exp(1j * self.rng.uniform(-np.pi, np.pi)).astype(np_complex_dtype)

        # Roll each axis
        nd = len(self.axes)
        input_torch = torch.roll(input, (shift,)*nd, dims=self.axes)

        # Move to sigpy
        dev = input_torch.device
        input_sigpy = input_torch.cpu().numpy()

        # Appoly random phase ...
        input_sigpy *= phase

        # Apply prox
        input_sigpy = self.W.H(sp.thresh.soft_thresh(self.lamda * alpha,
                                                        self.W(input_sigpy)))
        input_sigpy = input_sigpy.astype(np_complex_dtype)
        
        # Undo random phase ...
        input_sigpy *= np.conj(phase)
        
        # Move to pytorch
        output_torch = torch.asarray(input_sigpy).to(dev)

        # Unroll
        output_torch = torch.roll(output_torch, (-shift,)*nd, dims=self.axes)
        
        return output_torch

# FIXME TODO
class TV(nn.Module):
    def __init__(self,
                 im_size: tuple,
                 lamda: float,
                 norm: Optional[str] = 'l1',
                 max_iter: int = 50,
                 tol: float = 1e-4):
        """
        TV operator is defined as

        TV(x) = norm(Dx)

        Parameters:
        -----------
        im_size : tuple
            the image/volume dimensions
        lamda : float
            Regularization strength

        norm : str
            the type of norm to use from:
            ['l1', 'l2'] (Currently only 'l2' is implemented)
        
        max_iter : int
            Maximum number of iterations for the proximal operator
        
        tol : float
            Tolerance for convergence in the proximal operator
        """
        super().__init__()
        assert len(im_size) == 2 or len(im_size) == 3, 'Only 2D and 3D images are supported'
        self.im_size = im_size
        self.lamda = lamda
        self.norm = norm
        self.max_iter = max_iter
        self.tol = tol
    
    def forward(self,
                input: torch.Tensor,
                alpha: Optional[float] = 1.0):
        """
        Proximal operator for TV regularization using Chambolle's algorithm.

        Parameters:
        -----------
        input : torch.Tensor
            Image/volume input with shape (..., *im_size)
        alpha : float
            Proximal weighting term on g(x)

        Returns:
        --------
        output : torch.Tensor
            Proximal output
        """

        # Handle complex inputs by working with magnitude
        # For complex images, we apply TV to the magnitude and preserve phase
        is_complex = torch.is_complex(input)
        if is_complex:
            x_mag = torch.abs(input)
            x_phase = torch.angle(input)
        else:
            x_mag = input
            x_phase = None

        # Determine spatial dimensions based on im_size
        spatial_dims = len(self.im_size)
        if spatial_dims == 2:
            # 2D case: im_size = (H, W)
            # Input can be (H, W), (batch, H, W), or (..., H, W)
            # Extract spatial dimensions
            if x_mag.dim() == 2:
                # (H, W)
                spatial_shape = x_mag.shape
            else:
                # (..., H, W) - extract last 2 dimensions
                spatial_shape = x_mag.shape[-2:]
            # Use real dtype for dual variables (always real since we work with magnitude)
            # x_mag is always real (from torch.abs), so just use its dtype
            p = torch.zeros((2, *spatial_shape), dtype=x_mag.dtype, device=x_mag.device)
        elif spatial_dims == 3:
            # 3D case: im_size = (D, H, W)
            # Input can be (D, H, W), (batch, D, H, W), or (..., D, H, W)
            if x_mag.dim() == 3:
                # (D, H, W)
                spatial_shape = x_mag.shape
            else:
                # (..., D, H, W) - extract last 3 dimensions
                spatial_shape = x_mag.shape[-3:]
            # Use real dtype for dual variables (always real since we work with magnitude)
            # x_mag is always real (from torch.abs), so just use its dtype
            p = torch.zeros((3, *spatial_shape), dtype=x_mag.dtype, device=x_mag.device)
        else:
            raise ValueError('Only 2D and 3D images are supported.')

        tau = 0.25  # Step size

        # Work with magnitude for TV computation
        x_div = x_mag

        # Iteratively update dual variables
        for _ in range(self.max_iter):
            p_old = p.clone()

            # Compute gradient of divergence
            div_p = self.divergence(p)
            grad = self.gradient(x_div - alpha * self.lamda * div_p)

            # Update dual variables
            p = p + tau * grad

            # Projection onto the L2 unit ball
            # p is always real since we work with magnitude
            if self.norm == 'l2':
                norm_p = torch.sqrt(torch.sum(p ** 2, dim=0, keepdim=True))
                norm_p = torch.clamp(norm_p, min=1.0)
                p = p / norm_p
            elif self.norm == 'l1':
                p = torch.clamp(p, min=-1.0, max=1.0)
            else:
                raise ValueError("Unsupported norm type. Use 'l1' or 'l2'.")

            # Check convergence
            dp = p - p_old
            if dp.abs().max() < self.tol:
                break

        # Compute the output
        div_p = self.divergence(p)
        x_new_mag = x_mag - alpha * self.lamda * div_p
        
        # Reconstruct complex output if input was complex
        if is_complex:
            x_new = x_new_mag * torch.exp(1j * x_phase)
        else:
            x_new = x_new_mag

        return x_new

    def gradient(self, x):
        """
        Compute the gradient of x.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with spatial dimensions matching im_size

        Returns:
        --------
        grad : torch.Tensor
            Gradient of x with shape (ndirs, *spatial_shape)
        """
        spatial_dims = len(self.im_size)
        if spatial_dims == 2:
            # 2D case: x has shape (H, W) or (..., H, W)
            if x.dim() == 2:
                # (H, W)
                grad_x = F.pad(x[1:, :] - x[:-1, :], (0, 0, 0, 1), mode='constant', value=0)
                grad_y = F.pad(x[:, 1:] - x[:, :-1], (0, 1, 0, 0), mode='constant', value=0)
            else:
                # (..., H, W) - work on last 2 dimensions
                grad_x = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1), mode='constant', value=0)
                grad_y = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0), mode='constant', value=0)
            grad = torch.stack((grad_x, grad_y), dim=0)
        elif spatial_dims == 3:
            # 3D case: x has shape (D, H, W) or (..., D, H, W)
            if x.dim() == 3:
                # (D, H, W)
                grad_x = F.pad(x[1:, :, :] - x[:-1, :, :], (0, 0, 0, 0, 0, 1), mode='constant', value=0)
                grad_y = F.pad(x[:, 1:, :] - x[:, :-1, :], (0, 0, 0, 1, 0, 0), mode='constant', value=0)
                grad_z = F.pad(x[:, :, 1:] - x[:, :, :-1], (0, 1, 0, 0, 0, 0), mode='constant', value=0)
            else:
                # (..., D, H, W) - work on last 3 dimensions
                grad_x = F.pad(x[..., 1:, :, :] - x[..., :-1, :, :], (0, 0, 0, 0, 0, 1), mode='constant', value=0)
                grad_y = F.pad(x[..., :, 1:, :] - x[..., :, :-1, :], (0, 0, 0, 1, 0, 0), mode='constant', value=0)
                grad_z = F.pad(x[..., :, :, 1:] - x[..., :, :, :-1], (0, 1, 0, 0, 0, 0), mode='constant', value=0)
            grad = torch.stack((grad_x, grad_y, grad_z), dim=0)
        else:
            raise ValueError('Only 2D and 3D images are supported.')
        return grad

    def divergence(self, p):
        """
        Compute the divergence of p.

        Parameters:
        -----------
        p : torch.Tensor
            Dual variable with shape (ndirs, *spatial_shape)

        Returns:
        --------
        div : torch.Tensor
            Divergence of p with shape matching spatial dimensions
        """
        spatial_dims = len(self.im_size)
        if spatial_dims == 2:
            # p has shape (2, H, W)
            p_x = p[0]
            p_y = p[1]

            div_x = F.pad(p_x[:-1, :] - p_x[1:, :], (0, 0, 1, 0), mode='constant', value=0)
            div_y = F.pad(p_y[:, :-1] - p_y[:, 1:], (1, 0, 0, 0), mode='constant', value=0)
            div = div_x + div_y
        elif spatial_dims == 3:
            # p has shape (3, D, H, W)
            p_x = p[0]
            p_y = p[1]
            p_z = p[2]

            div_x = F.pad(p_x[:-1, :, :] - p_x[1:, :, :], (0, 0, 0, 0, 1, 0), mode='constant', value=0)
            div_y = F.pad(p_y[:, :-1, :] - p_y[:, 1:, :], (0, 0, 0, 1, 0, 0), mode='constant', value=0)
            div_z = F.pad(p_z[:, :, :-1] - p_z[:, :, 1:], (1, 0, 0, 0, 0, 0), mode='constant', value=0)
            div = div_x + div_y + div_z
        else:
            raise ValueError('Only 2D and 3D images are supported.')
        return div


class LocallyLowRank(nn.Module):
    """Version of LLR based on https://pmc.ncbi.nlm.nih.gov/articles/PMC10081201/

    Language based on spatiotemporal blocks
    """
    def __init__(
            self,
            input_size: Tuple,
            hparams: LLRHparams,
            input_type: Optional[Callable]= None,
    ):
        super().__init__()
        self.input_type = input_type if input_type is not None else complex_dtype
        self.hparams = hparams

        # Using a fixed random number generator so that recons are consistent
        self.rng = np.random.default_rng(1000)
        self.rnd_shift = hparams.rnd_shift

        # Derived
        self.block = Block(self.hparams.block_size, self.hparams.block_stride)
        self.block_weights = nn.Parameter(
            self.block.precompute_normalization(input_size).type(self.input_type),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor):
        """
        x: [N A H W [D]]
          - N: Batch dim
          - A: Temporal (subspace) dim
          - H, W, [D]: spatial dims

        """
        assert x.dim() >= 4
        block_dim = len(self.block.block_size)

        # Random shift 
        shift = round(self.rng.uniform(-self.rnd_shift, self.rnd_shift))

        # Roll in each axis by some shift amount
        x = torch.roll(x, (shift,)*block_dim, dims=tuple(range(-block_dim, 0)))

        # Extract Blocks
        x, nblocks = self.block(x)

        # Combine within-block dimensions
        # Move temporal dimension to be second-to-last
        unblocked_shape = x.shape # Save block shape for later
        x = rearrange(x, 'n a b ... -> n b a (...)')

        # Take SVD
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)

        # Threshold
        S = S - self.hparams.threshold
        S[S < 0] = 0.
        S = S.type(U.dtype)

        # Recompose blocks
        x = U @ torch.diag_embed(S) @ Vh

        # Unblock and normalize
        x = rearrange(x, 'n b a x -> n a b x')
        x = x.reshape(*unblocked_shape)
        x = self.block.adjoint(x, nblocks, norm_weights=self.block_weights)

        # Undo the roll in each shift direction
        x = torch.roll(x, (-shift,)*block_dim, dims=tuple(range(-block_dim, 0)))

        # Return the thresholded input
        return x

    def forward_mrf(self, x: torch.Tensor):
        """Simple wrapper that fixes dimensions
        x: [A H W [D]]

        Adds batch dim
        """
        assert x.dim() == 3 or x.dim() == 4
        x = x[None, ...]
        x = self(x)
        x = x[0, ...]
        return x