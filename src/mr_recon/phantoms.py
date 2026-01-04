import os
import torch
import numpy as np
import sigpy as sp

from typing import Optional
from einops import einsum, rearrange
from mr_recon.dtypes import real_dtype, complex_dtype
from mr_recon.utils import gen_grd, three_rotation_matrix, apply_window, np_to_torch
from mr_recon.fourier import fft, ifft

class shepp_logan:

    def __init__(self,
                 device: Optional[torch.device] = torch.device('cpu'),
                 dtype: Optional[torch.dtype] = complex_dtype):
        """
        Sheep logan model for generating 2D or 3D phantom and k-space data.
        Mostly copied from sigpy

        Parameters:
        -----------
        device : torch.device
            Device to store the phantom
        dtype : torch.dtype
            Data type of the phantom
        """

        self.device = device
        self.dtype = dtype

        # Store elipsoid params
        self.amps = torch.tensor([1, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                           device=device, dtype=real_dtype)
        self.scales = 1/torch.tensor([
            [.6900, .920, .810], # white big
            [.6624, .874, .780], # gray big
            [.1100, .310, .220], # right black
            [.1600, .410, .280], # left black
            [.2100, .250, .410], # gray center blob
            [.0460, .046, .050],
            [.0460, .046, .050],
            [.0460, .046, .050], # left small dot
            [.0230, .023, .020], # mid small dot
            [.0230, .023, .020]],
            device=device, dtype=real_dtype) * 2
        self.centers = torch.tensor([
            [0., 0., 0],
            [0., -.0184, 0],
            [.22, 0., 0],
            [-.22, 0., 0],
            [0., .35, -.15],
            [0., .1, .25],
            [0., -.1, .25],
            [-.08, -.605, 0],
            [0., -.606, 0],
            [.06, -.605, 0]],
            device=device, dtype=real_dtype) / 2
        self.angles = torch.deg2rad(torch.tensor([
            [0, 0, 0],
            [0, 0, 0],
            [-18, 0, -10],
            [18, 0, 30],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]],
            device=device, dtype=real_dtype))

    def img(self,
            im_size: tuple) -> torch.Tensor:
        """
        Generates shepp logan phantom of size im_size

        Parameters:
        -----------
        im_size : tuple
            Size of the phantom, 2 or 3 dims allowed

        img : torch.Tensor
            Shepp logan phantom with shape (im_size)
        """
        img = torch.zeros(im_size, device=self.device, dtype=self.dtype)
        r = gen_grd(im_size).type(real_dtype).to(self.device)
        for n in range(len(self.amps)):
            if (n == 5 or n == 6) and len(im_size) == 2:
                continue
            img += self.amps[n] * self.ellipsoid(r, self.angles[n], self.centers[n], self.scales[n])
            
        return img

    def ksp(self,
            trj: torch.Tensor) -> torch.Tensor:
        """
        Generates k-space data of shepp logan phantom

        Parameters:
        -----------
        trj : torch.Tensor
            Trajectory in k-space with shape (..., d)
        
        Returns:
        --------
        ksp : torch.Tensor
            k-space data with shape (...)
        """

        ksp = torch.zeros(trj.shape[:-1], device=self.device, dtype=self.dtype)
        for n in range(len(self.amps)):
            if (n == 5 or n == 6) and trj.shape[-1] == 2:
                continue
            ksp += self.amps[n] * self.ellipsoid_FT(trj, self.angles[n], self.centers[n], self.scales[n])
        return ksp
    
    def multi_chan_ksp(self,
                       trj: torch.Tensor,
                       mps: torch.Tensor,
                       kern_width: Optional[int] = 10) -> torch.Tensor:
        """
        Generates multi-channel k-space data of shepp logan phantom

        Parameters:
        -----------
        trj : torch.Tensor
            Trajectory in k-space with shape (..., d)
        mps : torch.Tensor
            Multi-channel maps with shape (c, *im_size)
        
        Returns:
        --------
        ksp : torch.Tensor
            k-space data with shape (c, ...)
        """
        # Consts
        im_size = mps.shape[1:]
        ndim = len(im_size)
        kern_size = (kern_width,) * ndim
        assert trj.shape[-1] == ndim

        # Start by fourier transforming maps to get small k-space kernels
        mps_ksp = fft(mps, dim=tuple(range(-ndim, 0)))
        tup = [slice(None),] + [slice(im_size[i]//2 - (kern_width//2), im_size[i]//2 + (kern_width//2)) for i in range(len(im_size))]
        mps_ksp_flipped = mps_ksp[tup].flip(dims=tuple(range(-ndim, 0)))
        mps_ksp_flipped = apply_window(mps_ksp_flipped, ndim, 'hamming')
        mps_ksp_flipped = rearrange(mps_ksp_flipped, 'c ... -> c (...)')

        # Coordinates of kernel of sensivity amps
        k_kern = gen_grd(kern_size, kern_size).type(real_dtype).to(self.device)
        k_kern = rearrange(k_kern, '... ndim -> (...) ndim')

        # Inner product flipped kernels with k-space
        trj_with_kern = trj[..., None, :] + k_kern
        ksp_phantom = self.ksp(trj_with_kern) # ... K d
        ksp = einsum(ksp_phantom, mps_ksp_flipped, '... K, c K -> c ...')

        return ksp

    @staticmethod
    def unit_sphere(r: torch.Tensor) -> torch.Tensor:
        """
        Analytical 3D/2D unit sphere/circle with radius 1

        Args:
        -----
        r : torch.Tensor
            3D coordinates with shape (..., 3 or 2)

        Returns:
        --------
        img : torch.Tensor
            Image values with shape (...)
        """
        return 1 * (r.norm(dim=-1) <= 1)

    @staticmethod
    def ellipsoid(r: torch.Tensor,
                theta: torch.Tensor,
                center: torch.Tensor,
                scales: torch.Tensor) -> torch.Tensor:
        """
        Model for ellipsoid e(r) given unit sphere u(r) is:

        e(r) = u(S(Rr - c))

        where 
        - R = Rx(theta_x) * Ry(theta_y) * Rz(theta_z) is the rotation matrix
        - c is the center of the ellopsoid
        - S = diag(s1, s2, s3) is the scaling matrix
        
        Args:
        -----
        r : torch.Tensor
            image domain coordinates with shape (..., 3)
        theta : torch.Tensor
            Rotation angles with shape (3,)
        center : torch.Tensor
            Center of ellipsoid with shape (3,)
        scale : torch.Tensor
            Scaling factors with shape (3,)
        
        Returns:
        --------
        img : torch.Tensor
            image values with shape (...)
        """
        if r.shape[-1] == 3:
            R = three_rotation_matrix(theta)
        elif r.shape[-1] == 2:
            theta_new = torch.zeros_like(theta)
            theta_new[-1] = theta[-1]
            R = three_rotation_matrix(theta_new)[:2, :2]
            center = center[:2]
            scales = scales[:2]
        r_rot = einsum(R, r, 'o i, ... i -> ... o')
        return shepp_logan.unit_sphere(scales * (r_rot - center))

    @staticmethod  
    def unit_sphere_FT(k: torch.Tensor) -> torch.Tensor:
        """
        Analytical 3D FT of unit sphere with radius 1

        Args:
        -----
        k : torch.Tensor
            k-space coordinatess with shape (..., 3 or 2)

        Returns:
        --------
        ksp : torch.Tensor
            k-space values with shape (...)
        """
        s = k.norm(dim=-1) * 2 * torch.pi
        if k.shape[-1] == 3:
            ksp = 4 * torch.pi * (torch.sin(s) - s * torch.cos(s)) / (s ** 3)
            ksp[s == 0] = 4 * torch.pi / 3
        elif k.shape[-1] == 2:
            ksp = 2 * torch.pi * torch.special.bessel_j1(s) / s
            ksp[s == 0] = 2 * torch.pi / 2 # or 3?
        return ksp

    @staticmethod
    def ellipsoid_FT(trj: torch.Tensor,
                    theta: torch.Tensor,
                    center: torch.Tensor,
                    scales: torch.Tensor) -> torch.Tensor:
        """
        Model for ellipsoid given in ellipsoid() function
        
        Args:
        -----
        trj : torch.Tensor
            k-space coordinates with shape (..., 3)
        theta : torch.Tensor
            Rotation angles with shape (3,)
        center : torch.Tensor
            Center of ellipsoid with shape (3,)
        scale : torch.Tensor
            Scaling factors with shape (3,)
        
        Returns:
        --------
        ksp : torch.Tensor
            k-space values with shape (...)
        """

        # Apply rotation to trajectory
        if trj.shape[-1] == 3:
            R = three_rotation_matrix(theta)
        elif trj.shape[-1] == 2:
            theta_new = torch.zeros_like(theta)
            theta_new[-1] = theta[-1]
            R = three_rotation_matrix(theta_new)[:2, :2]
            center = center[:2]
            scales = scales[:2]
        trj_rot = einsum(R, trj, 'o i, ... i -> ... o')

        # Phase due to offsets
        phase = einsum(trj_rot, center, '... o, o -> ...')
        ksp = torch.exp(-2j * torch.pi * phase)

        # Rotated & scaled FT of sphere
        ksp *= shepp_logan.unit_sphere_FT(trj_rot / scales) / torch.prod(scales)

        return ksp

class quant_phantom:
    
    def __init__(self,
                 isotropic_fov: Optional[bool] = True,
                 axial_slice: Optional[int] = None):
        """
        Parameters:
        -----------
        isotropic_fov : bool, optional
            If True, the tissue masks will be zero padded to have isotropic FOV.
        axial_slice : int, optional
            If not None, the tissue masks will be sliced along the axial direction.
        """
        filename = os.path.join(os.path.dirname(__file__), '../../download/single_comp/single_comp.npz')
        all_data = np.load(filename, allow_pickle=True)
        self.quant_maps = {
            't1': all_data['t1'],
            't2': all_data['t2'],
            't2*': all_data['t2*'],
            'pd': all_data['pd'],
            't1w': all_data['t1w'],
            't2w': all_data['t2w'],
            'pdw': all_data['pdw'],
        }
        for key in self.quant_maps.keys():
            # Zero pad to have isotropic FOV
            if isotropic_fov:
                self.quant_maps[key] = sp.resize(self.quant_maps[key], (max(self.quant_maps[key].shape),)*3)

            # Reshape from Z Y X to X Y Z
            self.quant_maps[key] = rearrange(self.quant_maps[key], 'z y x -> x y z')

            # Select a axial slice
            if axial_slice is not None:
                self.quant_maps[key] = self.quant_maps[key][..., axial_slice]
            
            # To Torch
            self.quant_maps[key] = np_to_torch(self.quant_maps[key])
    
    def t2star(self):
        return self.quant_maps['t2*']

    def t1(self):
        return self.quant_maps['t1']
    
    def t2(self):
        return self.quant_maps['t2']
    
    def pd(self):
        return self.quant_maps['pd']
    
    def t1_weighted(self):
        return self.quant_maps['t1w']
    
    def t2_weighted(self):
        return self.quant_maps['t2w']
    
    def pd_weighted(self):
        return self.quant_maps['pdw']

class multi_comp_phantom:
    
    def __init__(self,
                 isotropic_fov: Optional[bool] = True,
                 axial_slice: Optional[int] = None):
        """
        Parameters:
        -----------
        isotropic_fov : bool, optional
            If True, the tissue masks will be zero padded to have isotropic FOV.
        axial_slice : int, optional
            If not None, the tissue masks will be sliced along the axial direction.
        """

        # Load Data
        filename = os.path.join(os.path.dirname(__file__), '../../download/multi_comp/multi_comp.npz')
        all_data = np.load(filename, allow_pickle=True)
        keys = list(all_data.keys())[:-1]

        # Save as dicts
        self.tissue_masks = {}
        for key in keys:
            # Load tissue mask
            tissue_mask = all_data[key]

            # Zero pad to have isotropic FOV
            if isotropic_fov:
                tissue_mask = sp.resize(tissue_mask, (max(tissue_mask.shape),)*3)

            # Reshape from Z Y X to X Y Z
            tissue_mask = rearrange(tissue_mask, 'z y x -> x y z')

            # Select a axial slice
            if axial_slice is not None:
                tissue_mask = tissue_mask[..., axial_slice]
            
            self.tissue_masks[key] = np_to_torch(tissue_mask)
        self.tissue_props = all_data['tissue_props'].item()
    
    def weighted_img(self,
                     TR: float, 
                     TE: float, 
                     TI: Optional[float] = None, 
                     spin_echo: Optional[bool] = True):
        """
        Generates a mix of T1/T2/pd contrast. If inversion time TI
        is given, includes inversion recovery contrast.

        Parameters:
        -----------
        TR : float
            Repetition time in ms
        TE : float
            Echo time in ms
        TI : float, optional
            Inversion time in ms
        spin_echo : bool, optional
            If True, the sequence is spin echo and hence has T2 decay, otherwise T2*.
        
        Returns:
        --------
        img : torch.Tensor
            The generated contrast weighted image
        """

        img = None
        for key in self.tissue_props.keys():
            # Grab params and mask
            t1, t2, t2s, pd = self.tissue_props[key]
            if t1 == 0 or t2 == 0:
                continue
            msk = self.tissue_masks[key]

            # Spin echo case
            if spin_echo is None:
                E2 = np.exp(-TE / t2s)
            else:
                E2 = np.exp(-TE / t2)

            # Inv recov vs sat recov seq
            ER = np.exp(-TR / t1)
            if TI is None:
                msk *= pd * (1 - ER) * E2
            else:
                EI = np.exp(-TI / t1)
                msk *= pd * (1 - 2 * EI + ER) * E2
            
            # Append to image
            if img is None:
                img = msk
            else:
                img += msk
                
        return img
