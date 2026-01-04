import torch

from tqdm import tqdm
from typing import Optional

def gfactor_SENSE_PMR(R_ref: callable,
                      R_acc: callable,
                      ksp_ref: torch.Tensor,
                      ksp_acc: torch.Tensor,
                      noise_var: Optional[float] = 1e-2,
                      n_replicas: Optional[int] = 100,
                      verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Calculates the g-factor map of a SENSE reconstruction using Psuedo Multiple Replica method.
    """
    
    var_ref = calc_variance_PMR(R_ref, ksp_ref, noise_var, n_replicas, verbose)
    var_acc = calc_variance_PMR(R_acc, ksp_acc, noise_var, n_replicas, verbose)
        
    # Calculate g-factor directly
    # Add small epsilon to prevent division by zero and ensure non-negative ratio
    eps = 1e-12
    ratio = var_acc / (var_ref + eps)
    gfactor = torch.sqrt(torch.clamp(ratio, min=0.0))
    
    return gfactor

def gfactor_SENSE_diag(AHA_inv_ref: callable,
                       AHA_inv_acc: callable,
                       inp_example: torch.Tensor,
                       n_replicas: Optional[int] = 100,
                       sigma: Optional[float] = 1e-2,
                       rnd_vec_type: Optional[str] = None,
                       verbose: Optional[bool] = True) -> torch.Tensor:
    """
    Calculates the g-factor map of a SENSE reconstruction using diagonal estimation method.
    
    Parameters:
    ----------
    AHA_inv_ref : callable
        Reference inverse gram operator
    AHA_inv_acc : callable
        Accelerated inverse gram operator
    inp_example : torch.Tensor
        Example input to AHA_inv_ref/AHA_inv_acc, helps in determining the shape, device, and dtype of the input.
    n_replicas : int
        Number of replicas to use.
    rnd_vec_type : str
        Type of random vectors to use. Can be 'real' or 'complex'. Defaults to datatype of input.
    verbose : bool
        Toggles progress bar
    
    Returns:
    --------
    gfactor : torch.Tensor
        g-factor map with same size as image.    
    """
    
    diag_ref = diagonal_estimator(AHA_inv_ref, inp_example, n_replicas, rnd_vec_type,sigma=sigma, verbose=verbose)
    diag_acc = diagonal_estimator(AHA_inv_acc, inp_example, n_replicas, rnd_vec_type,sigma=sigma, verbose=verbose)
    gfactor = (diag_acc / diag_ref).sqrt()
    return gfactor

def calc_variance_PMR(R: callable,
                      ksp: torch.Tensor,
                      noise_var: Optional[float] = 1e-2,
                      n_replicas: Optional[int] = 100,
                      verbose: Optional[bool] = True) -> torch.Tensor:
    
    """
    Computes variance map of reconstruction R(noise) using Psuedo Mulitple Replica method:
    
    var = \sum_n |R(noise_n)|^2 / (N * noise_var)
    
    where noise_n is i.i.d. Gaussian noise with variance noise_var.
    
    Robson PM, et. al. Comprehensive quantification of signal-to-noise ratio and g-factor for image-based and k-space-based parallel imaging reconstructions. Magn Reson Med. 2008 Oct;60(4):895-907. doi: 10.1002/mrm.21728. PMID: 18816810; PMCID: PMC2838249.
    
    Parameters:
    ----------
    R : callable
        Linear reconsruction operator mapping from k-space to image domain.
    ksp : torch.Tensor
        k-space data (can be zeros)
    noise_var : float
        Variance of k-space pseudo-noise 
    n_replicas : int
        Number of replicas to use.
    verbose : bool
        Toggles progress bar
        
    Returns: 
    --------
    var : torch.Tensor
        variance map with same size as image.
    """
    
    if ksp.norm() < 1e-9:
        # If k-space is empty, noise is the only signal.
        # Use a running sum to avoid storing all replicas in memory.
        var = None
        pbar = tqdm(range(n_replicas), 'PMR Loop (zero-ksp)', disable=not verbose)
        for n in pbar:
            noise = (noise_var ** 0.5) * torch.randn_like(ksp)
            recon = R(ksp + noise)
            var_n = recon.abs() ** 2 / (n_replicas * noise_var)
            if var is None:
                var = var_n
            else:
                var += var_n
    else:
        recons = []
        pbar = tqdm(range(n_replicas), 'PMR Loop', disable=not verbose)
        for n in pbar:
            noise = (noise_var ** 0.5) * torch.randn_like(ksp)
            recons.append(R(ksp + noise))
        
        recons = torch.stack(recons, dim=0)
        # Variance = E[|X|^2] - |E[X]|^2 = Var(Real) + Var(Imag)
        # torch.var calculates the sample variance (unbiased by default)
        # For complex input, torch.var calculates var(real) + var(imag) which is what we want for total noise power
        var = torch.var(recons, dim=0, unbiased=False) / noise_var
        
    return var

def diagonal_estimator(M: callable,
                      inp_example: torch.Tensor,
                      n_replicas: Optional[int] = 100,
                      rnd_vec_type: Optional[str] = 'complex',
                      sigma: Optional[float] = 1.0,
                      verbose: Optional[bool] = True,
                      scale: Optional[float] = 1.0) -> torch.Tensor:
    """
    Estimates the diagoal elements of some matrix operator M using a modified Hutchinson's method.
    
    Parameters:
    ----------
    M : callable
        Square linear operator
    inp_example : torch.Tensor
        Example input to M, helps in determining the shape, device, and dtype of the input.
    n_replicas : int
        Number of replicas to use.
    rnd_vec_type : str
        Type of random vectors to use. Can be 'real' or 'complex'. Defaults to datatype of input.
    verbose : bool
        Toggles progress bar
        
    Returns:
    --------
    diag : torch.Tensor
        Estimated diagonal elements of M.
    
    Dharangutte, Prathamesh, and Christopher Musco. "A tight analysis of hutchinson's diagonal estimator." Symposium on Simplicity in Algorithms (SOSA). Society for Industrial and Applied Mathematics, 2023.
    """
    # Get constants from example input 
    idtype = inp_example.dtype
    idevice = inp_example.device
    ishape = inp_example.shape
    
    # Random vector generators
    rnd_vec_comp = lambda : torch.exp(1j * 2 * torch.pi * torch.rand(ishape, device=idevice)).type(idtype)
    rnd_vec_real = lambda : 2 * torch.randint(0, 2, ishape, device=idevice).type(idtype) - 1
    rnd_vec_comp_gauss = lambda : torch.randn(ishape, device=idevice).type(idtype) + 1j * torch.randn(ishape, device=idevice).type(idtype)
    
    # Function to generate random vectors
    if 'real' in rnd_vec_type:
        rnd_vec = rnd_vec_real
    elif 'gauss' in rnd_vec_type:
        rnd_vec = rnd_vec_comp_gauss
    else:
        rnd_vec = rnd_vec_comp
    
    # Estimate diagonal
    diag = torch.zeros_like(inp_example)
    if verbose:
        pbar = tqdm(range(n_replicas), 'Diagonal Estimation')
    else:
        pbar = range(n_replicas)
        
    for i in pbar:
        v = rnd_vec()
        Mv = M(v)
        diag += (v.conj() * Mv) / n_replicas
    diag = diag
    return diag.abs()


def incremental_calc_variance_PMR(R, ksp, noise_var, N_values, verbose=True):
    """
    Incrementally calculates the variance map of a reconstruction using the Pseudo Multiple Replica method,
    yielding the result at each N specified in N_values.

    This is more efficient than calling calc_variance_PMR for each N, as it reuses samples.
    """
    running_sum = 0
    cumulative_N = 0
    sorted_N = sorted(N_values)

    pbar = tqdm(total=sorted_N[-1], desc="Incremental PMR", disable=not verbose)
    for n_target in sorted_N:
        num_new_samples = n_target - cumulative_N
        for _ in range(num_new_samples):
            noise = (noise_var ** 0.5) * torch.randn_like(ksp)
            recon = R(ksp + noise)
            if isinstance(running_sum, int):
                 # Initialize running sum with the correct shape and type
                 running_sum = torch.zeros_like(recon, dtype=torch.float32)
            running_sum += recon.abs()**2
            pbar.update(1)
        
        cumulative_N = n_target
        # The variance is the expectation of |R(noise)|^2, which is the sum / N
        # The final division by noise_var is part of the PMR definition
        yield running_sum / (cumulative_N * noise_var)
    pbar.close()


def incremental_diagonal_estimator(M, inp_example, N_values, rnd_vec_type='complex', sigma=1.0, verbose=True):
    """
    Incrementally calculates the diagonal of an operator using Hutchinson's method,
    yielding the result at each N specified in N_values.

    This is more efficient than calling diagonal_estimator for each N, as it reuses samples.
    """
    # ... (Setup random vector generator as in diagonal_estimator)
    ishape, idevice, idtype = inp_example.shape, inp_example.device, inp_example.dtype
    if rnd_vec_type == 'complex':
        rnd_vec_comp = lambda : torch.exp(1j * 2 * torch.pi * torch.rand(ishape, device=idevice)).type(idtype)
        rnd_vec = lambda : rnd_vec_comp()
    elif rnd_vec_type == 'real':
        rnd_vec_real = lambda : torch.randn(ishape, device=idevice).sign().type(idtype)
        rnd_vec = lambda : rnd_vec_real()
    
    running_sum = 0
    cumulative_N = 0
    sorted_N = sorted(N_values)

    pbar = tqdm(total=sorted_N[-1], desc="Incremental Hutchinson", disable=not verbose)
    for n_target in sorted_N:
        num_new_samples = n_target - cumulative_N
        for _ in range(num_new_samples):
            v = rnd_vec() * sigma
            Mv = M(v)
            if isinstance(running_sum, int):
                running_sum = torch.zeros_like(v, dtype=torch.float32)
            running_sum += (v.conj() * Mv).real
            pbar.update(1)
            
        cumulative_N = n_target
        yield (running_sum / cumulative_N).abs()
    pbar.close()


def gfactor_sense(mps, Rx, Ry, l2_reg=0.0):
    """
    mps  : (C, H, W) complex128/complex64 coil-sensitivity maps
    Rx,Ry: in-plane acceleration factors
    """
    C, H, W = mps.shape
    R = Rx * Ry                                                     # #alias voxels

    # ------------------------------------------------------------------
    # 1. Build aliasing (coil × alias × H × W) sensitivity tensor
    # ------------------------------------------------------------------
    shifted = []
    for i in range(Rx):
        for j in range(Ry):
            shifted.append(torch.roll(
                mps, shifts=(i * H // Rx, j * W // Ry), dims=(-2, -1)))
    c_mat = torch.stack(shifted, dim=1)                             # (C,R,H,W)

    # ------------------------------------------------------------------
    # 2. Cᴴ C   ->   (H,W,R,R)
    # ------------------------------------------------------------------
    chc = torch.einsum('crhw,cshw->hwrs', c_mat.conj(), c_mat)      # r,s = alias

    # ------------------------------------------------------------------
    # 3. Invert per pixel with Tikhonov regularisation
    # ------------------------------------------------------------------
    eye = l2_reg * torch.eye(R, dtype=chc.dtype, device=mps.device)
    eye = eye.expand(H, W, R, R)
    chc_inv = torch.linalg.inv(chc + eye)                           # (H,W,R,R)

    # ------------------------------------------------------------------
    # 4. g-factor map (take alias 0 ≡ target voxel)
    #    g = √( (CᴴC)₀₀ · (CᴴC)⁻¹₀₀ )
    # ------------------------------------------------------------------
    diag_chc      = torch.diagonal(chc,     dim1=-2, dim2=-1)[..., 0]   # (H,W)
    diag_chc_inv  = torch.diagonal(chc_inv, dim1=-2, dim2=-1)[..., 0]   # (H,W)
    g_map = (diag_chc.real * diag_chc_inv.real).sqrt()                  # (H,W)

    return g_map.cpu()