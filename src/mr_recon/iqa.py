import numpy as np
import matplotlib.pyplot as plt

# If you want to compute SSIM, uncomment and install scikit-image
from skimage.metrics import structural_similarity as ssim

def compare_g_factors(g, std_monte, std_hutch, n_monte=100, n_monte_ref=300, time_g=None, time_monte=None, time_hutch=None, plot=True, recon_image=None):
    """
    Compare analytical g-factor map (g) with Monte Carlo and Hutchinson's method maps.

    Args:
        g (torch.Tensor): Analytical g-factor map (PyTorch tensor).
        std_monte (torch.Tensor): Monte Carlo standard deviation map (PyTorch tensor).
        std_hutch (torch.Tensor): Hutchinson's method standard deviation map (PyTorch tensor).
        n_monte (int, optional): Number of Monte Carlo noise iterations. Default is 100.
        plot (bool, optional): Whether to display the plot. Default is True.
        recon_image (torch.Tensor, optional): Reconstructed image to display. Default is None.

    Returns:
        dict: A dictionary containing MSE, NMSE, and Pearson correlation 
              for both Monte Carlo and Hutchinson's method.
    """
    # 1. Convert tensors to NumPy arrays
    g_np = g.cpu().numpy()
    std_monte_np = std_monte.cpu().numpy()
    std_hutch_np = std_hutch.cpu().numpy()
    
    if recon_image is not None:
        recon_image_np = recon_image.cpu().numpy()

    # 2. Flatten the images for metric calculations
    g_flat = g_np.flatten()
    std_monte_flat = std_monte_np.flatten()
    std_hutch_flat = std_hutch_np.flatten()

    # 3. Take non-zero values
    g_flat = g_flat[g_flat != 0]
    std_monte_flat = std_monte_flat[std_monte_flat != 0]
    std_hutch_flat = std_hutch_flat[std_hutch_flat != 0]

    # 4. Compute Normalized Root Mean Squared Error (NRMSE)
    mse_analytical_monte = np.sqrt(np.mean((g_flat - std_monte_flat) ** 2))
    mse_analytical_hutch = np.sqrt(np.mean((g_flat - std_hutch_flat) ** 2))

    nrmse_analytical_monte = 100*mse_analytical_monte / np.max(g_flat)
    nrmse_analytical_hutch = 100*mse_analytical_hutch / np.max(g_flat)

    # 6. Compute Pearson Correlation Coefficient
    corr_analytical_monte = np.corrcoef(g_flat, std_monte_flat)[0, 1] *100
    corr_analytical_hutch = np.corrcoef(g_flat, std_hutch_flat)[0, 1] *100

    # 7. (Optional) Compute Structural Similarity Index (SSIM)
    #    This requires 2D or 3D data (not flattened), and they must be normalized [0, 1].
    #    If you want to compute SSIM for the entire 2D maps, uncomment below.

    g_norm = (g_np - g_np.min()) / (g_np.max() - g_np.min())
    std_monte_norm = (std_monte_np - std_monte_np.min()) / (std_monte_np.max() - std_monte_np.min())
    std_hutch_norm = (std_hutch_np - std_hutch_np.min()) / (std_hutch_np.max() - std_hutch_np.min())

    ssim_monte = ssim(g_norm, std_monte_norm, data_range=1.0)
    ssim_hutch = ssim(g_norm, std_hutch_norm, data_range=1.0)

    #Compute time
    print(f"Time for Reference (MC): {time_g:.2f} seconds")
    print(f"Time for Monte Carlo: {time_monte:.2f} seconds")
    print(f"Time for Hutchinson's: {time_hutch:.2f} seconds")

    # 8. Print or return the similarity metrics
    print("=== Similarity Metrics ===")
    print(f"NRMSE % (Analytical vs. Monte Carlo): {nrmse_analytical_monte:.4f}")
    print(f"Pearson Correlation (Analytical vs. Monte Carlo): {corr_analytical_monte:.4f}\n")

    print(f"NRMSE % (Analytical vs. Hutchinson's): {nrmse_analytical_hutch:.4f}")
    print(f"Pearson Correlation (Analytical vs. Hutchinson's): {corr_analytical_hutch:.4f}\n")

    print(f"SSIM (Analytical vs. Monte Carlo): {ssim_monte:.4f}")
    print(f"SSIM (Analytical vs. Hutchinson's): {ssim_hutch:.4f}")

    # 9. Plot the maps if requested
    if plot:
        plt.figure(figsize=(18, 6))
        # plt.suptitle(f"G-Factor Comparison with N = {n_monte} Noise Iterations", fontsize=16)

        # (A) Analytical G-Factor Std
        if n_monte_ref is not None:
            plt.subplot(2, 3, 1)
            plt.title(f"Reference G-Factor derived from Monte Carlo with N={n_monte_ref}\nTime: {time_g:.2f} seconds")
            plt.imshow(g_np, cmap="jet", vmin=g_np.min(), vmax=g_np.max())
            plt.colorbar()
            plt.axis("off")
        else:
            plt.subplot(2, 3, 1)
            plt.title(f"Reference Analytical G-Factor \n Time: {time_g:.2f} seconds")
            plt.imshow(g_np, cmap="jet", vmin=g_np.min(), vmax=g_np.max())
            plt.colorbar()
            plt.axis("off")

        # Display reconstructed image
        if recon_image is not None:
            plt.subplot(2, 3, 4)
            plt.title("Reconstructed Image")
            plt.imshow(np.abs(recon_image_np), cmap="gray")
            plt.colorbar()
            plt.axis("off")

        # (B) Monte Carlo Std
        plt.subplot(2, 3, 2)
        plt.title(f"PMR G-Factor with N={n_monte}\nTime: {time_monte:.2f} seconds\nNRMSE: {nrmse_analytical_monte:.4f}%\nPearson Corr: {corr_analytical_monte:.4f}%")
        plt.imshow(std_monte_np, cmap="jet", vmin=g_np.min(), vmax=g_np.max())
        plt.colorbar()
        plt.axis("off")

        # (C) Hutchinson's Std
        plt.subplot(2, 3, 3)
        plt.title(f"Our method G-Factor with N={n_monte}\nTime: {time_hutch:.2f} seconds\nNRMSE: {nrmse_analytical_hutch:.4f}%\nPearson Corr: {corr_analytical_hutch:.4f}%")
        plt.imshow(std_hutch_np, cmap="jet", vmin=g_np.min(), vmax=g_np.max())
        plt.colorbar()
        plt.axis("off")


        # (D) Difference Monte Carlo
        plt.subplot(2, 3, 5)
        plt.title("Difference PMR (10x)")
        plt.imshow(10*np.abs(g_np - std_monte_np), cmap="jet", vmin=g_np.min(), vmax=np.max(g_np))
        plt.colorbar()
        plt.axis("off")

        # (E) Difference Hutchinson's
        plt.subplot(2, 3, 6)
        plt.title("Difference Our's (10x)")
        plt.imshow(10*np.abs(g_np - std_hutch_np), cmap="jet", vmin=g_np.min(), vmax=np.max(g_np))
        plt.colorbar()
        plt.axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # 10. Return metrics as a dictionary (optional)
    metrics = {
        "nrmse_analytical_monte": nrmse_analytical_monte,
        "nrmse_analytical_hutch": nrmse_analytical_hutch,
        "corr_analytical_monte": corr_analytical_monte,
        "corr_analytical_hutch": corr_analytical_hutch,
        # "ssim_analytical_monte": ssim_monte,
        # "ssim_analytical_hutch": ssim_hutch,
    }

    return metrics
