import torch
import numpy as np
import cvxpy as cp

from mr_recon.utils import np_to_torch
from typing import Optional
from einops import rearrange, einsum
from scipy import integrate
from scipy.interpolate import interp1d
from sigpy.mri import spiral
from sigpy.mri.rf import min_time_gradient

gamma_bar = 42.58e6 # Hz/T

def design_rewinder(grad: np.ndarray, 
                    t_rewind: float, 
                    g_max: float, 
                    s_max: float, 
                    dt_grad: float):
    """
    Appends a rewinder to the end of a gradient waveform.

    Parameters:
    -----------
    grad : np.ndarray
        Gradient waveform to append the rewinder to [G/cm]
        has shape (N, 2)
    t_rewind : float
        Rewinder waveform time [ms]
    g_max : float
        Max gradient amplitude [G/cm]
    s_max : float
        Max slew rate [G/cm/ms]
    dt_grad : float
        Gradient dwell time [ms]

    Returns:
    --------
    grad_new : np.ndarray
        New gradient waveform with the rewinder appended [G/cm]
        has shape (N + L, 2)
        where L = round(t_rewind / dt_grad)
    k_new : np.ndarray
        New k-space trajectory [cm^-1]
        has shape (N + L, 2)
    s_new : np.ndarray
        New slew rate waveform [G/cm/ms]
        has shape (N + L - 1, 2)
    """

    # Create rewinder at fixed length
    L = round(t_rewind / dt_grad)
    gcp = cp.Variable((L, 2))

    # Build constraints
    constraints = [
        gcp[0] == grad[-1], # Start at the end of the gradient
        gcp[-1, 0] == 0, # End at zero
        gcp[-1, 1] == 0, # End at zero
        cp.sum(gcp[:-1], axis=0) + (gcp[-1])/2 + \
        np.sum(grad[1:], axis=0) + grad[0] / 2 == 0.0, # End at center (0) k-space
        cp.sum(cp.square(gcp), axis=1) <= g_max ** 2, # Max gradient
        cp.sum(cp.square(cp.diff(gcp, axis=0)), axis=1) \
            <= (s_max * dt_grad) ** 2, # Max slew rate
    ]

    # Make sure the feasibility problem is DCP
    problem = cp.Problem(cp.Minimize(1), constraints) 
    assert problem.is_dcp(), 'Problem is not DCP, choose larger t_rewind_ms'

    # Solve
    problem.solve()
    assert problem.status == cp.OPTIMAL, f'Solution status is {problem.status}'

    # Append to gradient and return new waveforms
    grad_new = np.concatenate((grad, gcp.value), axis=0)
    k_new = integrate.cumtrapz(grad_new, initial=0, axis=0) * 4.257 * dt_grad
    s_new = np.diff(grad_new, axis=0) / dt_grad

    return grad_new, k_new, s_new

def design_spiral_trj(gmax, smax, res, fov, nshots=1, alpha=1.0, dt=2e-6, rewinder=False):
    
    # Conversions
    g_max_G_cm = gmax * 100
    s_max_G_cm_ms = smax / 10

    # Vardense spiral design
    k = spiral(fov,
               N=round(fov/res),
               f_sampling=0.1,
               R=1,
               ninterleaves=nshots,
               alpha=alpha,
               gm=gmax,
               sm=smax)
    k = k[:k.shape[0]//nshots]

    # Convert to 1/cm
    k = k / 100

    # time optimal design
    M = 4
    dt_grad = dt * 1e3 * M
    k_3d = np.concatenate((k, k[:, :1] * 0), axis=-1)
    g, k, _, _ = min_time_gradient(k_3d, 
                                    gmax=g_max_G_cm*0.99, 
                                    smax=s_max_G_cm_ms*0.99, 
                                    dt=dt_grad)
    g = g[..., :2]
    k = k[..., :2]

    # Rewind to center
    if rewinder:
        g, k, _ = design_rewinder(g, 
                                  t_rewind=1,
                                  g_max=g_max_G_cm,
                                  s_max=s_max_G_cm_ms,
                                  dt_grad=dt_grad)
    
    # Interpolate back MX
    t_orig = np.arange(g.shape[0])
    t_new = np.arange(0, g.shape[0] - 1, 1/M)
    g = interp1d(t_orig, g, axis=0, kind='cubic')(t_new)
    k = interp1d(t_orig, k, axis=0, kind='cubic')(t_new)
        
    # 2x2 rotation matrix by theta
    def rot2d(theta_deg):
        theta = np.deg2rad(theta_deg)
        return np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    
    # Rotate the waveforms
    gs = []
    ks = []
    for i in range(nshots):
        angle = i * 360 / nshots
        R = rot2d(angle)
        grot = einsum(R, g, 'twoout twoin, nro twoin -> nro twoout')
        krot = einsum(R, k, 'twoout twoin, nro twoin -> nro twoout')
        gs.append(grot)
        ks.append(krot)
    gs = np.stack(gs, axis=1)
    ks = np.stack(ks, axis=1)

    # Convert back to SI units
    trj = np_to_torch(ks) * 100
    gs = np_to_torch(gs) / 100

    return trj

def design_epi_trj(gmax, smax, res, fov_y, nshots=1, fov_z=0, mb=0, dt=2e-6):

    area_read = 1 / (res * gamma_bar)
    area_blip = nshots / (fov_y * gamma_bar)
    if mb != 0:
        area_blip_z = 1 / (fov_z * gamma_bar)
    else:
        area_blip_z = 0

    tx, gx = design_grad_blip(gmax, smax, area_read, dt)
    ty, gy = design_grad_blip(gmax, smax, area_blip, dt)
    gz = gy * area_blip_z / area_blip
    
    grads = torch.stack([gx, gx * 0, gx * 0], dim=-1)
    grads = torch.cat([grads, torch.stack([gy * 0, gy, gy * 0], dim=-1)], dim=0)
    grads[len(tx):, -1] = gz
    
    n = len(torch.arange(0, 1/res, nshots/fov_y))
    grads = torch.repeat_interleave(grads[None,], n, dim=0)
    grads[1::2, :, 0] = -grads[::2, :, 0][:grads.shape[0]//2, :]
    if mb != 0:
        grads[::mb, :, 2] *= -(mb - 1)
    grads = rearrange(grads, 'n t d -> (n t) d')[:-len(ty), :]

    trj = torch.cumulative_trapezoid(grads, dt * torch.arange(grads.shape[0]), dim=0) * gamma_bar
    trj[:, 2] -= trj[:, 2].mean()
    trj[:, 1] -= trj[:, 1].max()/2
    trj[:, 0] -= trj[:, 0].max()/2

    # Shots
    trj = torch.tile(trj[:, None, :], [1, nshots, 1])
    for i in range(nshots):
        trj[:, i, 1] += i / fov_y

    return trj

def rect(x):
    return 1.0 * (x >= 0) * (x < 1)

def inc_right_tri(x):
    return x * (x >= 0) * (x < 1)

def dec_right_tri(x):
    return (1 - x) * (x >= 0) * (x < 1)

def triangle(x):
    return inc_right_tri(2 * x) + dec_right_tri(2 * x - 1)

def design_grad_blip(gmax: float, 
                     smax: float, 
                     areas: torch.Tensor, 
                     dt: Optional[float] = 2e-6,
                     same_width: Optional[bool] = False) -> torch.Tensor:
    """
    Design a gradient blip with a given area

    Parameters:
    -----------
    gmax : float
        Maximum gradient amplitude [T/m]
    smax : float
        Maximum slew rate [T/m/s]
    areas : torch.Tensor
        Areas of the gradient blips with arb shape (...) [Ts/m]
    dt : float
        Time step [s]
    same_width : bool
        If True, all triangles will have the same width, and only the gradient 
        amplitude will change between them

    Returns:
    --------
    ts : torch.Tensor
        Time points with shape (num_time_points) [s]
    grad : torch.Tensor
        Gradient waveform with shape (..., num_time_points) [T/m]
    """

    # Flatten and handle float case
    if type(areas) == float:
        was_float = True
        areas = torch.tensor([areas])
    else:
        was_float = False
    areas_flt = areas.reshape((-1,))

    # Start with triangle pulse 
    ts, grad_tri, T_tri = calc_tri_blip(gmax, smax, areas_flt, dt)

    # Handle cases where the triangle is not big enough with trapezoid
    T_sq = torch.maximum((areas_flt - torch.trapz(grad_tri, dim=-1, dx=dt)).abs() / gmax, torch.ones_like(areas_flt) * dt)
    ts = torch.arange(0, ts[-1] + T_sq.max(), dt, device=areas.device)
    grad_tri = torch.cat([grad_tri, 
                          torch.zeros((*grad_tri.shape[:-1], len(ts) - grad_tri.shape[-1]), device=grad_tri.device)], 
                          dim=-1)
    T_tri = T_tri[:, None]
    T_sq = T_sq[:, None]
    p1 = gmax * inc_right_tri(ts / (T_tri / 2))
    p2 = gmax * rect((ts - (T_tri / 2)) / T_sq)
    p3 = gmax * dec_right_tri((ts - (T_tri / 2) - T_sq) / (T_tri / 2))
    grad_trap = (p1 + p2 + p3) * torch.sign(areas_flt)[:, None]
    indicator = 1 * (T_sq > dt)
    grad = grad_trap * indicator + grad_tri * (1 - indicator)

    # Same width case
    if same_width:
        i_max = torch.argmax(areas_flt.abs())
        grad = torch.repeat_interleave(grad[i_max, :][None, :], len(areas_flt), dim=0)

    # Rescale for trapz integration and reshape
    areas_grad = torch.trapz(grad, dim=-1, dx=dt)
    grad = grad * (areas_flt / areas_grad).unsqueeze(-1)
    grad_rs = grad.reshape((*areas.shape, len(ts)))

    # Handle zero area cases
    grad_rs = torch.nan_to_num(grad_rs, nan=0.0)

    if was_float:
        return ts, grad_rs[0]
    return ts, grad_rs

def calc_tri_blip(gmax: float,
                  smax: float, 
                  areas: torch.Tensor,
                  dt: Optional[float] = 2e-6,
                  same_width: Optional[bool] = False) -> torch.Tensor:
    """
    Calculates triangle gradient for a blip of given area

    *Note*
    If the area is too large, the triangle will simply represent 
    the largest possible araa (gmax ** 2 / smax)

    Parameters:
    -----------
    gmax : float
        Maximum gradient amplitude [T/m]
    smax : float
        Maximum slew rate [T/m/s]
    areas : torch.Tensor
        Areas of the gradient blips with arb shape (...) [Ts/m]
    dt : float
        Time step [s]
    same_width : bool
        If True, all triangles will have the same width, and only the gradient 
        amplitude will change between them

    Returns:
    --------
    ts : torch.Tensor
        Time points with shape (num_time_points) [s]
    grad : torch.Tensor
        Gradient waveform with shape (..., num_time_points) [T/m]
    T_tri : torch.Tensor
        Triangle width [s] with shape (...)
    """

    # Flatten and handle float case
    if type(areas) == float:
        was_float = True
        areas = torch.tensor([areas])
    else:
        was_float = False
    areas_flt = areas.reshape((-1,))

    # Assert max area is not exceeded
    max_area = (gmax ** 2) / smax
    areas_flt = torch.minimum(areas_flt.abs(), max_area * torch.ones_like(areas_flt)) * torch.sign(areas_flt)

    # Calculate triangle width
    T_tri = torch.minimum(2 * (areas_flt.abs() / smax).sqrt(), torch.ones_like(areas_flt) * 2 * gmax / smax)
    T_tri = torch.maximum(T_tri, torch.ones_like(T_tri) * dt)
    T_max = T_tri.max()
    if same_width:
        T_tri = torch.ones_like(T_tri, device=areas.device) * T_max

    # Grad = 2 * A / T, but limit to gmax
    G_tri = torch.minimum(2 * areas_flt.abs() / T_tri, torch.ones_like(T_tri) * gmax) * torch.sign(areas_flt)

    # Generate time points and gradient
    ts = torch.arange(0, T_max + dt, dt, device=areas.device)
    grad = triangle(ts / T_tri[:, None]) * G_tri[:, None]
    grad = grad * (areas_flt.abs() / torch.trapz(grad.abs(), dim=-1, dx=dt)).unsqueeze(-1)
    grad = grad.reshape((*areas.shape, len(ts)))

    # Handle zero area cases
    grad = torch.nan_to_num(grad, nan=0.0)

    # Return
    if was_float:
        return ts, grad[0], T_tri[0]
    return ts, grad, T_tri
