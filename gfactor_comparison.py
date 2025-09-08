import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from torch.func import vmap, jvp
import time
from einops import rearrange
import matplotlib.gridspec as gridspec

# Assuming the following libraries are in the python path
from sigpy.mri import poisson
from mr_recon.fourier import gridded_nufft
from mr_recon.utils import np_to_torch, gen_grd
from mr_recon.algs import FISTA, density_compensation, power_method_operator
from mr_recon.prox import TV
from mr_recon.linops import sense_linop, batching_params


def fista_reconstruct(kspace, A, proxg, max_eigen, num_iters=30):
    """Performs FISTA reconstruction."""
    AHb = A.adjoint(kspace)
    AHb = AHb / (max_eigen ** 0.5)
    AHA = lambda x: A.normal(x) / max_eigen
    img_recon = FISTA(AHA, AHb, proxg, num_iters=num_iters, verbose=False)
    return img_recon


def monte_carlo_variance_incremental(ksp, A, proxg, max_eigen, max_n, n_values_to_store, sigma, num_iters=20):
    """
    Computes Monte Carlo variance, storing results at specified intermediate N values.
    """
    device = ksp.device
    results = {}
    time_results = {}
    imgs_all_cpu = []
    
    start_time = time.time()
    for n in tqdm(range(max_n), desc=f'Monte Carlo (max_n={max_n})'):
        noise_real = torch.randn_like(ksp.real)
        noise_imag = torch.randn_like(ksp.imag)
        ksp_noise = ksp + (noise_real + 1j * noise_imag) * sigma
        
        img_recon = fista_reconstruct(ksp_noise, A, proxg, max_eigen, num_iters=num_iters)
        imgs_all_cpu.append(img_recon.cpu())
        
        if (n + 1) in n_values_to_store:
            elapsed_time = time.time() - start_time
            time_results[n + 1] = elapsed_time

            # Batched variance calculation to avoid OOM
            n_current = len(imgs_all_cpu)
            batch_size = 50

            sum_imgs = torch.zeros_like(imgs_all_cpu[0], device=device, dtype=torch.complex64)
            for i in range(0, n_current, batch_size):
                batch = torch.stack(imgs_all_cpu[i:i+batch_size]).to(device)
                sum_imgs += batch.sum(dim=0)
            mean = sum_imgs / n_current

            sum_sq_diff = torch.zeros_like(imgs_all_cpu[0], device=device, dtype=torch.float32)
            for i in range(0, n_current, batch_size):
                batch = torch.stack(imgs_all_cpu[i:i+batch_size]).to(device)
                sum_sq_diff += ((batch - mean).abs()**2).sum(dim=0)
            
            variance = sum_sq_diff / (n_current - 1) if n_current > 1 else torch.zeros_like(sum_sq_diff)
            
            std_img = torch.sqrt(variance)
            std_img /= sigma
            variance = std_img.pow(2)
            results[n + 1] = variance
            
    print(f"Monte Carlo (max_n={max_n}) took {time.time() - start_time:.2f} seconds.")
    return results, time_results


def hutchinson_variance_incremental(ksp, A, proxg, max_eigen, max_n, n_values_to_store, input_noise, num_iters=20):
    """
    Computes Hutchinson variance incrementally to save memory, storing results at specified N values.
    """
    device = ksp.device
    dtype = ksp.dtype
    results = {}
    time_results = {}
    
    def model_func(kspace_input):
        return fista_reconstruct(kspace_input, A, proxg, max_eigen, num_iters=num_iters)

    def compute_variance_sample(v):
        _, u = jvp(model_func, (ksp,), (v,))
        return u.real.pow(2) + u.imag.pow(2)

    # Process samples incrementally and store on CPU
    variance_samples_cpu = []
    start_time = time.time()
    
    for i in tqdm(range(max_n), desc=f"Hutchinson (max_n={max_n})"):
        v_shape = (1,) + ksp.shape
        real_part = (torch.randint(0, 2, v_shape, device=device).float() * 2 - 1)
        imag_part = (torch.randint(0, 2, v_shape, device=device).float() * 2 - 1)
        v_sample = (real_part + 1j * imag_part).to(dtype)
        
        # Compute one sample and store on CPU
        variance_sample = vmap(compute_variance_sample)(v_sample)
        variance_samples_cpu.append(variance_sample.cpu())

        if (i + 1) in n_values_to_store:
            current_samples = torch.cat(variance_samples_cpu, dim=0).to(device)
            avg_variance = current_samples.mean(dim=0)
            # avg_variance *= input_noise ** 2 # This was the bug - G-factor should be independent of noise level
            results[i + 1] = avg_variance
            time_results[i + 1] = time.time() - start_time
            
    print(f"Hutchinson (max_n={max_n}) took {time.time() - start_time:.2f} seconds.")
    return results, time_results


def calculate_metrics(g_ref, g_test):
    """Calculates NRMSE and Pearson Correlation."""
    g_ref_np = g_ref.cpu().numpy()
    g_test_np = g_test.cpu().numpy()

    g_ref_flat = g_ref_np.flatten()
    g_test_flat = g_test_np.flatten()

    # Use only non-zero values from the reference for calculation
    mask = g_ref_flat != 0
    g_ref_flat = g_ref_flat[mask]
    g_test_flat = g_test_flat[mask]
    
    # NRMSE
    mse = np.sqrt(np.mean((g_ref_flat - g_test_flat) ** 2))
    nrmse = 100 * mse / np.max(g_ref_flat) if np.max(g_ref_flat) > 0 else 0

    # Pearson Correlation
    corr = np.corrcoef(g_ref_flat, g_test_flat)[0, 1] * 100
    return nrmse, corr


def plot_n_comparison(g_ref, pmr_results, hutch_results, n_values, pmr_times, hutch_times, n_ref):
    """Plots the comparison of G-factor methods for different N."""
    num_n = len(n_values)
    fig = plt.figure(figsize=(10, 5 * (num_n + 1)))
    gs = gridspec.GridSpec(num_n + 1, 2, figure=fig)
    
    fig.suptitle('G-Factor Estimation vs. Number of Trials (N)', fontsize=20)

    g_ref_np = g_ref.cpu().numpy()
    vmin = g_ref_np.min()
    vmax = g_ref_np.max()

    # --- Reference Image Row ---
    ax_ref = fig.add_subplot(gs[0, :])
    im = ax_ref.imshow(g_ref_np, cmap='jet', vmin=vmin, vmax=vmax)
    ax_ref.set_title(f'Reference G-Factor (PMR with N={n_ref})', fontsize=14)
    ax_ref.axis('off')

    for i, n in enumerate(n_values):
        # --- PMR column ---
        ax_pmr = fig.add_subplot(gs[i + 1, 0])
        pmr_map = pmr_results[n]
        time_pmr = pmr_times[n]
        nrmse_pmr, corr_pmr = calculate_metrics(g_ref, pmr_map)
        ax_pmr.imshow(pmr_map.cpu().numpy(), cmap='jet', vmin=vmin, vmax=vmax)
        ax_pmr.set_title(f'PMR (N={n})\nTime: {time_pmr:.2f}s, NRMSE: {nrmse_pmr:.2f}%, Corr: {corr_pmr:.2f}%')
        ax_pmr.axis('off')

        # --- Hutchinson column ---
        ax_hutch = fig.add_subplot(gs[i + 1, 1])
        hutch_map = hutch_results[n]
        time_hutch = hutch_times[n]
        nrmse_hutch, corr_hutch = calculate_metrics(g_ref, hutch_map)
        ax_hutch.imshow(hutch_map.cpu().numpy(), cmap='jet', vmin=vmin, vmax=vmax)
        ax_hutch.set_title(f'Our Method (N={n})\nTime: {time_hutch:.2f}s, NRMSE: {nrmse_hutch:.2f}%, Corr: {corr_hutch:.2f}%')
        ax_hutch.axis('off')

    # --- Color Bar ---
    fig.colorbar(im, ax=fig.get_axes(), shrink=0.8, location='right')
    
    gs.tight_layout(fig, rect=[0, 0.03, 0.95, 0.95], h_pad=4)
    plt.savefig("gfactor_N_comparison.png", dpi=300)
    plt.show()


def main():
    """Main execution function."""
    # --- Setup ---
    print("Setting up reconstruction problem...")
    torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    slice_ = 20
    
    with h5py.File('file1000000.h5', 'r') as f:
        img = torch.tensor(f['reconstruction_rss'], device=torch_dev)[slice_]
        mps = torch.tensor(f['jsense-12-cf=4']['maps'], device=torch_dev)
    
    mps = mps[slice_, :, :, :, 0]
    im_size = (img.shape[0], img.shape[1])
    C = mps.shape[-1]
    mps = rearrange(mps, 'h w c -> c h w')

    R = 10
    mask = poisson(im_size, accel=R, calib=(24, 24), tol=0.3, crop_corner=False)
    mask = np_to_torch(mask.real).type(torch.float32).to(torch_dev)
    inds = torch.argwhere(mask > 0)
    trj = gen_grd(im_size, im_size).to(torch_dev)
    trj = trj[inds[:, 0], inds[:, 1]]
    nufft = gridded_nufft(im_size)
    bparams = batching_params(C)
    dcf = density_compensation(trj, im_size)
    A = sense_linop(im_size, trj, mps, dcf, nufft=nufft, bparams=bparams)
    ksp = A(img)
    
    x0 = torch.randn_like(img)
    _, max_eigen = power_method_operator(A.normal, x0)
    max_eigen *= 1.01
    proxg = TV(im_size=img.shape[-2:], lamda=1e-7, norm='l2')
    print("Setup complete.")

    # --- Experiment Parameters ---
    N_values = [5, 10, 20, 40, 80, 120]
    N_ref = 1000
    sigma = 1e-5
    num_iters = 20

    # --- Generate Data ---
    ref_results, _ = monte_carlo_variance_incremental(ksp, A, proxg, max_eigen, N_ref, [N_ref], sigma, num_iters)
    g_ref = ref_results[N_ref]

    pmr_results, pmr_times = monte_carlo_variance_incremental(ksp, A, proxg, max_eigen, max(N_values), N_values, sigma, num_iters)

    hutch_results, hutch_times = hutchinson_variance_incremental(ksp, A, proxg, max_eigen, max(N_values), N_values, sigma, num_iters)

    # --- Plot Results ---
    print("Plotting comparison...")
    plot_n_comparison(g_ref, pmr_results, hutch_results, N_values, pmr_times, hutch_times, N_ref)
    print("Comparison plot saved as gfactor_N_comparison.png")

if __name__ == "__main__":
    main()
