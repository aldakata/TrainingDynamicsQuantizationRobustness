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
    With the PCA computed, I will sample new weights to cover as much of the top2 components PCA space.
    I will save this samples to different file_paths.
    Then, I will load a different sample on the top2 PCA space, project back to the 160M parameter space, and evaluate the loss there.
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
DST = f"{DST}/stable_decay_stable_decay"
os.makedirs(DST, exist_ok=True)
paths = [
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_0/ckpt_step_4000.pth",  "brown", "Decay 10%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_0/ckpt_step_8000.pth",  "brown", "Decay 10%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_0/ckpt_step_12000.pth", "brown", "Decay 10%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_0/ckpt_step_14000.pth", "brown", "Decay 10%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_0/ckpt_step_18000.pth", "brown", "Decay 10%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_0/ckpt_step_20000.pth", "brown", "Decay 10%"),

    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_1/ckpt_step_38000.pth",  "coral", "Decay 20%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_1/ckpt_step_42000.pth",  "coral", "Decay 20%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_1/ckpt_step_46000.pth",  "coral", "Decay 20%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_1/ckpt_step_48000.pth",  "coral", "Decay 20%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_1/ckpt_step_52000.pth",  "coral", "Decay 20%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_1/ckpt_step_54000.pth",  "coral", "Decay 20%"),
    
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_2/ckpt_step_72000.pth",  "orange", "Decay 30%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_2/ckpt_step_76000.pth",  "orange", "Decay 30%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_2/ckpt_step_80000.pth",  "orange", "Decay 30%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_2/ckpt_step_82000.pth",  "orange", "Decay 30%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_2/ckpt_step_86000.pth",  "orange", "Decay 30%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_2/ckpt_step_88000.pth",  "orange", "Decay 30%"),
    
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_3/ckpt_step_106000.pth", "red", "Decay 40%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_3/ckpt_step_110000.pth", "red", "Decay 40%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_3/ckpt_step_114000.pth", "red", "Decay 40%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_3/ckpt_step_116000.pth", "red", "Decay 40%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_3/ckpt_step_120000.pth", "red", "Decay 40%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_3/ckpt_step_122000.pth", "red", "Decay 40%"),
    
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_4/ckpt_step_140000.pth", "pink", "Decay 50%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_4/ckpt_step_144000.pth", "pink", "Decay 50%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_4/ckpt_step_148000.pth", "pink", "Decay 50%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_4/ckpt_step_150000.pth", "pink", "Decay 50%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_4/ckpt_step_154000.pth", "pink", "Decay 50%"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_03/job_idx_4/ckpt_step_156000.pth", "pink", "Decay 50%"),

    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_173500.pth", "purple", "Decay F"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_177000.pth", "purple", "Decay F"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_179000.pth", "purple", "Decay F"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_182500.pth", "purple", "Decay F"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_187000.pth", "purple", "Decay F"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_cooldown_04/job_idx_0/ckpt_step_190000.pth", "purple", "Decay F"),

    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_2000.pth",     "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_12000.pth",    "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_18000.pth",    "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_30000.pth",    "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_36000.pth",    "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_46000.pth",    "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_52000.pth",   "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_58000.pth",   "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_70000.pth",   "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_78000.pth",    "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_86000.pth",    "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_96000.pth",    "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_106000.pth",   "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_116000.pth",   "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_124000.pth",   "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_134000.pth",  "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_baseline_01/job_idx_0/ckpt_step_144000.pth",  "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_resume_01/job_idx_0/ckpt_step_154000.pth",    "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_resume_01/job_idx_0/ckpt_step_164000.pth",     "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_resume_01/job_idx_0/ckpt_step_173500.pth",     "olive", "Stable"),
    ("/fast/atatjer/scalinglawsquantization/checkpoints/pythia_160M_100BT_resume_01/job_idx_0/ckpt_step_182000.pth",     "olive", "Stable"),
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
        flat_params = torch.cat([p.detach().flatten() for p in model.parameters()]) # Only if param regquires grad, otherwise it doesn't make sense to change it.
        vector_list.append(flat_params)
        del model
        del tmp
        print(f"{i}/{len(paths)} Model loaded and flattened")

    X = torch.stack(vector_list) # (N,P)
    X = torch.tensor(X, dtype=torch.float32, device='cuda') # (N,P)
    mean = X.mean(0, keepdim=True) # (,P)
    Xc = X - mean # (N,P)
    C = (Xc @ Xc.T) / (Xc.shape[1] - 1) # (N) = (N,P)@(P,N)
    U, S, Vt = torch.linalg.svd(C) # (N,N), (N), (N,N)
    explained_variance = S**2 / (Xc.shape[0] - 1) # (N)
    explained_variance_ratio = explained_variance / explained_variance.sum() # (N)
    print(sum(explained_variance_ratio[:2]), explained_variance, explained_variance_ratio)
    del Vt, explained_variance
    eigvecs_topk = U # (N, k)
    S_topk = torch.sqrt(S) # (K)
    V = (Xc.T @ eigvecs_topk) # (P,K) = (P,N)@(N,K)
    V = V/ S_topk 
    projs = Xc @ V # (N,K) = (N,P)@(P,K)
    X_recon = projs @ V.T + mean  # (N, P)
    projs = projs.to('cpu')
    torch.save({
        'mean': mean.squeeze(0),  # (160M,)
        'topk_param_space': V,           # (160M, K)
        'explained_variance_ratio': explained_variance_ratio,
        'projections': projs,
    }, f'{DST}/pca_topk.pt')
else:
    state_dict = torch.load(f'{DST}/pca_topk.pt')
    projs = state_dict['projections']
plotting_meta = [
    ([36, 0, 1, 2, 3, 4, 5],                                                                                    'coral', 'Decay 0'),
    ([36, 37, 38, 39, 40, 6, 7, 8, 9, 10, 11],                                                                  'orange', 'Decay 1'),
    ([36, 37, 38, 39, 40, 41, 42, 43, 44, 12, 13, 14, 15, 16, 17],                                              'blue', 'Decay 2'),
    ([36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 18, 19, 20, 21, 22, 23],                              'brown', 'Decay 3'),
    ([36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 24, 25, 26, 27, 28, 29],              'red', 'Decay 4'),
    ([36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 30, 31, 32, 33, 34, 35],  'purple', 'Decay F'),
    ([36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56 ],                     'olive', 'Stable'),
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

# Sample from the Top2 PCA grid.
# Sampling strategy is gaussian around the weights, and then Latin Hypercube Sampling on the rest of the cube.
# Sampling from the projection space
def gaussian_sample_neighbourhood(X, sigma, n_samples):
    rows,cols = X.shape
    samples = np.zeros((rows*n_samples, cols))
    for i in range(n_samples):
        Z = np.random.randn(rows, cols)*sigma
        samples[rows*i:rows*(i+1), :] = X+Z
    return samples
around_ckpts = gaussian_sample_neighbourhood(projs, 1.e5, 2)

from scipy.stats import qmc
sampler = qmc.LatinHypercube(d=2, strength=2)
lhs_samples = sampler.random(n=289) # must be the square of a prime number
l_bounds = [-1.25e7, -0.9e7]
u_bounds = [1e7, 0.9e7]
lhs_samples = qmc.scale(lhs_samples, l_bounds, u_bounds)

projs_arr = projs.numpy()

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

lhs_samples_x = lhs_samples[:, 0]
lhs_samples_y = lhs_samples[:, 1]
ax.scatter(
    lhs_samples_x,
    lhs_samples_y,
    color='black',
    marker='*',
)

# around_ckpts = gaussian_sample_neighbourhood(projs, 1.e6, 1)
# around_ckpts_x = around_ckpts[:, 0]
# around_ckpts_y = around_ckpts[:, 1]
# ax.scatter(
#     around_ckpts_x,
#     around_ckpts_y,
#     color='gray',
#     marker='x',
# )

ax.legend()
ax.set_title("Training trajectory in PCA space")
plt.savefig(f"{DST}/landscape_sampled.png")
total_samples = np.vstack((projs_arr, lhs_samples))
total_samples = torch.tensor(total_samples, dtype=torch.float32)
print(f"Total samples {total_samples.shape}")
torch.save({
    'total_samples': total_samples,
}, f'{DST}/pca_top2_samples.pt')

# # Requires many loss evaluations. Current val_set is  ~600kTKs, let's get an estimate for this, otherwise, it is not that terrible to have a run time of 2 minutes per each
# # considering that this can be parallelized for each of the models easily leading to less than 2k evaluations total, which is doable in the cluster.
# # this will serve as a template for an abstracted, config input, sweepable script.
# device = 'cuda'
# ds = load_from_disk('/fast/atatjer/hf_fast/datasets/refinedweb/')

# num_workers=8
# block_size=2048
# batch_size=20
# tokenised = ds.map(
#     tokenize,
#     num_proc=num_workers,
#     remove_columns=ds.column_names
#     )
# concat_ids = [tok for ex in tokenised["ids"] for tok in ex]
# blocks = [concat_ids[i:i+block_size+1]           # +1 for the label shift
#         for i in range(0, len(concat_ids) - (block_size+1), block_size)]

# dl = torch.utils.data.DataLoader(utils.BlockDataset(blocks),
#                 batch_size=batch_size,
#                 shuffle=False,
#                 pin_memory=True)

# model = None
# evaluation.eval(model, dl, device)