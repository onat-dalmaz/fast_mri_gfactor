# MRI g-factor & SNR efficiency maps for modern iterative recon

Code and notebooks for estimating **spatially varying noise amplification** (g-factor / 1/g SNR efficiency) in **parallel MRI reconstructions**, with an emphasis on **iterative** (CG-SENSE) and **non-Cartesian** settings.

The goal is practical: make voxel-wise noise/SNR maps easier to compute and compare across reconstruction choices.

---

## What’s in this repo

- **Notebooks** demonstrating g-factor / 1/g estimation in:
  - **Cartesian CG-SENSE (knee)** with an **analytical reference** for validation
  - **Non-Cartesian spiral CG-SENSE (phantom)** with a **high-N PMR surrogate reference**
- Core utilities for:
  - Operator-style recon code (forward/adjoint)
  - diagonal estimation via image-space probing
  - Baseline **Pseudo Multiple Replica (PMR)** comparisons

---

## Key idea (high-level)

Traditional **PMR** estimates noise by:
1) adding random noise to k-space many times,  
2) reconstructing each replica,  
3) measuring voxel-wise variance from the ensemble.

This is general but can be expensive because it repeats full reconstructions.

This project explores a complementary approach:
- estimate the **diagonal of the reconstructed-image covariance** using **stochastic probing** (Hutchinson-style estimators),
- implemented using **matrix–vector products** that reuse existing solver primitives (e.g., CG with \(A\) / \(A^H\)).

The notebooks focus on comparing **convergence behavior** and practical runtime/quality trade-offs against PMR.

---

## Demos (Jupyter notebooks)

### 1) Non-Cartesian spiral phantom (regularized CG-SENSE)

![Non-Cartesian Phantom Convergence](notebooks/assets/compressed/cartesian_knee_convergence_small.gif)

**Notebook:** `notebooks/non_cartesian_phantom_gfactor_comparison.ipynb`

- Multi-coil GRE spiral phantom + measured trajectory  
- Retrospective shot subsampling (e.g., keep every \(R\)-th interleave)  
- Regularized CG-SENSE with NUFFT/DCF and Toeplitz normal-op acceleration  
- Reference: high-replica PMR surrogate (e.g., \(N_\mathrm{ref}=10{,}000\))

Assets (if included in repo):
- `notebooks/assets/noncartesian_phantom_convergence.gif`
- `notebooks/assets/compressed/noncartesian_phantom_convergence_small.gif`

---

### 2) Cartesian knee (unregularized CG-SENSE, analytical reference)
**Notebook:** `notebooks/cartesian_knee_gfactor_comparison.ipynb`

- Stanford knee dataset (multi-coil Cartesian)  
- Retrospective uniform undersampling (e.g., \(R=2\) along phase-encode)  
- Unregularized CG-SENSE  
- Reference: closed-form analytical SENSE g-factor

Assets (if included in repo):
- `notebooks/assets/cartesian_knee_convergence.gif`
- `notebooks/assets/compressed/cartesian_knee_convergence_small.gif`

---

## Quick start

### Setup
```bash
git clone https://github.com/<your-org-or-user>/<your-repo>.git
cd <your-repo>

conda env create -f environment.yml
conda activate <env-name>
```

If needed:
```bash
pip install h5py einops
```

### Run notebooks
```bash
jupyter notebook
```

Open:
- `notebooks/non_cartesian_phantom_gfactor_comparison.ipynb`
- `notebooks/cartesian_knee_gfactor_comparison.ipynb`

---

## Notes on metrics & visualization

- Many figures show **\(1/g\)** rather than \(g\), since \(1/g\) is directly interpretable as **SNR efficiency**.
- Convergence is typically evaluated vs:
  - an **analytical reference** (Cartesian), or
  - a **high-N PMR surrogate reference** (non-Cartesian / no closed form).

---

## References (background)

- Pruessmann et al., SENSE (parallel imaging / g-factor)
- Robson et al., SNR and g-factor quantification in parallel imaging
- Hutchinson (stochastic trace/diagonal estimation), and follow-on diagonal estimation work (e.g., Bekas)

---

## Citation / use

This is research code intended for reproducible demonstrations. If you use ideas or figures, please cite the associated manuscript / preprint and/or relevant prior work listed above.
