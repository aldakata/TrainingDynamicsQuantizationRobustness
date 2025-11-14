"""
    Given a training trajectory this script computes the vectors between all the checkpoints and the final model.
    We use PCA to get two principal orthogonal axes.
    We plot the training trajectory on the projected dimensions to see if the trajectory can be visualized in an interpretable way.
    If so, we compute the validation loss (probably a subset of the original validation loss, to have more evaluations for the same compute)
    and use these to plot a matplotlib contour to visualize the loss geometry around this points.
    Jupyternotebook-esque style.
"""

"""
    This code computes the PCA of a list of training trajectories with plotting heuristics.
    With the PCA computed, I will sample new weights to cover as much of the topk components PCA space.
    I will save this samples to different file_paths.
    Then, I will load a different sample on the topk PCA space, project back to the 160M parameter space, and evaluate the loss there.
    These loss evaluations will define the contour of the loss.
    Duplicate the number of samples to have more info for the different decays and have a smoother transition from stable to decay.
    From 20 to 57
"""
from filelock import SoftFileLock
import filelock
filelock.FileLock = SoftFileLock

import os
import torch
from copy import deepcopy
from transformers import AutoConfig, AutoModelForCausalLM
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import evaluation
from datasets import load_dataset, load_from_disk

DST = "/fast/atatjer/finetuningrobustness/landscapes"
DST = f"{DST}/stable_decay"
os.makedirs(DST, exist_ok=True)
paths = [
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_0/ckpt_step_4000.pth",  "brown", "Decay 10%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_0/ckpt_step_8000.pth",  "brown", "Decay 10%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_0/ckpt_step_12000.pth", "brown", "Decay 10%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_0/ckpt_step_14000.pth", "brown", "Decay 10%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_0/ckpt_step_18000.pth", "brown", "Decay 10%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_0/ckpt_step_20000.pth", "brown", "Decay 10%"),

    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_1/ckpt_step_38000.pth",  "coral", "Decay 20%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_1/ckpt_step_42000.pth",  "coral", "Decay 20%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_1/ckpt_step_46000.pth",  "coral", "Decay 20%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_1/ckpt_step_48000.pth",  "coral", "Decay 20%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_1/ckpt_step_52000.pth",  "coral", "Decay 20%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_1/ckpt_step_54000.pth",  "coral", "Decay 20%"),
    
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_2/ckpt_step_72000.pth",  "orange", "Decay 30%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_2/ckpt_step_76000.pth",  "orange", "Decay 30%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_2/ckpt_step_80000.pth",  "orange", "Decay 30%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_2/ckpt_step_82000.pth",  "orange", "Decay 30%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_2/ckpt_step_86000.pth",  "orange", "Decay 30%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_2/ckpt_step_88000.pth",  "orange", "Decay 30%"),
    
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_3/ckpt_step_106000.pth", "red", "Decay 40%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_3/ckpt_step_110000.pth", "red", "Decay 40%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_3/ckpt_step_114000.pth", "red", "Decay 40%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_3/ckpt_step_116000.pth", "red", "Decay 40%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_3/ckpt_step_120000.pth", "red", "Decay 40%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_3/ckpt_step_122000.pth", "red", "Decay 40%"),

    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_4/ckpt_step_140000.pth", "pink", "Decay 50%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_4/ckpt_step_144000.pth", "pink", "Decay 50%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_4/ckpt_step_148000.pth", "pink", "Decay 50%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_4/ckpt_step_150000.pth", "pink", "Decay 50%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_4/ckpt_step_154000.pth", "pink", "Decay 50%"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_4/ckpt_step_156000.pth", "pink", "Decay 50%"),

    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_173500.pth", "purple", "Decay F"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_177000.pth", "purple", "Decay F"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_179000.pth", "purple", "Decay F"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_182500.pth", "purple", "Decay F"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_187000.pth", "purple", "Decay F"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_190000.pth", "purple", "Decay F"),

    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_2000.pth",     "olive", "Stable"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_12000.pth",    "olive", "Stable"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_18000.pth",    "olive", "Stable"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_30000.pth",    "olive", "Stable"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_36000.pth",    "olive", "Stable"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_46000.pth",    "olive", "Stable"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_52000.pth",   "olive", "Stable"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_58000.pth",   "olive", "Stable"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_70000.pth",   "olive", "Stable"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_78000.pth",    "olive", "Stable"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_86000.pth",    "olive", "Stable"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_96000.pth",    "olive", "Stable"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_106000.pth",   "olive", "Stable"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_116000.pth",   "olive", "Stable"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_124000.pth",   "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_134000.pth",  "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_144000.pth",  "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_resume_01/job_idx_0/ckpt_step_154000.pth",    "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_resume_01/job_idx_0/ckpt_step_164000.pth",     "olive", "Stable"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_resume_01/job_idx_0/ckpt_step_173500.pth",     "olive", "Stable"),
    # ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_resume_01/job_idx_0/ckpt_step_182000.pth",     "olive", "Stable"),
]

# Load and flatten to get the data matrix (N,P), where N is the number of samples and P is the dimention of each, in my case (15, 160M)
if not f'{DST}/pca_topk.pt' or True:
    model_cfg = AutoConfig.from_pretrained(f"EleutherAI/pythia-160m")
    base_model =  AutoModelForCausalLM.from_config(model_cfg)

    vector_list = []
    for i, (path, _, labels) in enumerate(paths):
        model = deepcopy(base_model)
        state_dict = torch.load(path, map_location='cpu')
        tmp = model.load_state_dict(state_dict['state_dict'])
        flat_params = torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])
        vector_list.append(flat_params)
        del model
        del tmp
        print(f"{i}/{len(paths)} Model loaded and flattened")
    
    # The model converged at index 5. We will compute the PCA of the diretions to convergence
    X = torch.stack(vector_list).to(torch.float64)
    X = X-X[5]

    # Method 1:
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    projs = U * S
    X_recon = projs @ Vt + mean

    explained_variance = S**2 / (Xc.shape[0] - 1) # (N)
    explained_variance_ratio = explained_variance / explained_variance.sum() # (N)
    print(sum(explained_variance_ratio[:2]), explained_variance, explained_variance_ratio)
    print(X-X_recon)

    torch.save({
        'mean': mean,  # (160M,)
        'topk_param_space': Vt,           # (160M, 2)
        'explained_variance_ratio': explained_variance_ratio,
        'projections': projs,
    }, f'{DST}/pca_topk.pt')
else:
    state_dict = torch.load(f'{DST}/pca_topk.pt')
    projs = state_dict['projections']

plotting_meta = [
    ([6, 7, 8, 9, 0, 1, 2, 3, 4, 5,],  'purple', 'Decay F'),
    ([6, 7, 8, 9],                     'olive', 'Stable'),
]
fig, ax = plt.subplots(figsize=(9, 6))
for indices, color, label in plotting_meta:
    x_proj = projs[indices, 0]
    y_proj = projs[indices, 1]
    ax.plot(
        x_proj,
        y_proj,
        color=color,
        label=label,
        marker='o'
    )

ax.legend()
ax.set_title("Training trajectory in PCA space")
plt.savefig(f"{DST}/landscape.png")

def gaussian_sample_neighbourhood(X, sigma, n_samples):
    rows,cols = X.shape
    samples = torch.zeros((rows*n_samples, cols))
    for i in range(n_samples):
        Z = torch.randn((rows, cols))*sigma
        samples[rows*i:rows*(i+1), :] = X+Z
    return samples
# print('Guassian sampling')
# around_ckpts = torch.tensor(gaussian_sample_neighbourhood(Xc, 3e-4, 2), dtype=torch.float32)
# around_ckpts_proj = around_ckpts @ Vt

projs_arr = projs[:, :2].numpy()


mins = projs_arr.min(axis=0)
maxs = projs_arr.max(axis=0)

# from scipy.stats import qmc
# sampler = qmc.LatinHypercube(d=2, strength=2)
# lhs_samples = sampler.random(n=289) # must be the square of a prime number
# lhs_samples = qmc.scale(lhs_samples, mins, maxs)

# Create 1D arrays for each axis
x = np.linspace(mins[0], maxs[0], 20)
y = np.linspace(mins[1], maxs[1], 15)

# Create 2D grid
X, Y = np.meshgrid(x, y, indexing='ij')

fig, ax = plt.subplots(figsize=(9, 6))
for indices, color, label in plotting_meta:
    x_proj = projs[indices, 0]
    y_proj = projs[indices, 1]
    ax.plot(
        x_proj,
        y_proj,
        color=color,
        label=label,
        marker='o'
    )

points = np.stack([X.ravel(), Y.ravel()], axis=1)  # shape: [num_points, 2]
ax.scatter(
    points[:, 0],
    points[:, 1],
    color='black',
    marker='*'
)

# lhs_samples_x = lhs_samples[:, 0]
# lhs_samples_y = lhs_samples[:, 1]
# ax.scatter(
#     lhs_samples_x,
#     lhs_samples_y,
#     color='black',
#     marker='*',
# )

ax.legend()
ax.set_title("Training trajectory in PCA space")
plt.savefig(f"{DST}/landscape_sampled.png")
print(f"Total samples {points.shape}")
torch.save({
    'meshgrid': points,
}, f'{DST}/pca_topk_meshgrid.pt')