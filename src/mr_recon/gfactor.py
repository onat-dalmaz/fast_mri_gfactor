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
    
    Parameters:
    ----------
    R_ref : callable
        Reference linear reconsruction operator mapping from k-space to image domain
    R_acc : callable
        Accelerated linear reconsruction operator mapping from k-space to image domain
    ksp_ref : torch.Tensor
        Reference k-space data (can be zeros)
    ksp_acc : torch.Tensor
        Accelerated k-space data (can be zeros)
    noise_var : float
        Variance of k-space pseudo-noise
    n_replicas : int
        Number of replicas to use.
    verbose : bool
        Toggles progress bar
        
    Returns:
    --------
    gfactor : torch.Tensor
        g-factor map with same size as image.
    """
    
    var_ref = calc_variance_PMR(R_ref, ksp_ref, noise_var, n_replicas, verbose)
    var_acc = calc_variance_PMR(R_acc, ksp_acc, noise_var, n_replicas, verbose)
        
    # Calculate g-factor directly
    gfactor = torch.sqrt(var_acc / var_ref)
    
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
    Comptutes variance map of reconstruction R(noise) using Psuedo Mulitple Replica method:
    
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
        var = None
        for n in tqdm(range(n_replicas), 'PMR Loop', disable=not verbose):
            noise = (noise_var ** 0.5) * torch.randn_like(ksp * 0)
            
            recon = R(ksp + noise)
            var_n = recon.abs() ** 2 / (n_replicas * noise_var)
            if var is None:
                var = var_n
            else:
                var += var_n
    else:
        recons = []
        for n in tqdm(range(n_replicas), 'PMR Loop', disable=not verbose):
            noise = (noise_var ** 0.5) * torch.randn_like(ksp * 0)
            if n < 2:
                print(noise.norm())
            recons.append(R(ksp + noise))
        recons = torch.stack(recons, dim=0)
        var = torch.var(recons, dim=0)
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
    for n in tqdm(range(n_replicas), 'Diagonal Estimator Loop', disable=not verbose):
        v = (rnd_vec())*sigma
        if n == 0:
            print(f"norm of v: {torch.norm(v)}")
            print(f"shape of v: {v.shape}")
        Mv = M(v)
        diag += (v.conj() * Mv) / n_replicas
    diag = diag
    return diag.abs()

def gfactor_sense(mps, Rx, Ry, l2_reg=0.0, device='cuda'):
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
    eye = l2_reg * torch.eye(R, dtype=chc.dtype, device=device)
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