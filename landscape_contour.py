from filelock import SoftFileLock
import filelock
# Override FileLock globally to use SoftFileLock
filelock.FileLock = SoftFileLock

from absl import app, flags, logging
from collections import defaultdict

from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForCausalLM
from copy import deepcopy
from datasets import load_dataset, load_from_disk
import torch, math
import wandb

import utils
import evaluation
import importer

import os

flags.DEFINE_integer('job_idx', 0, 'Job idx for job-array sweeps. From 0 to n-1.')
FLAGS = flags.FLAGS

def main(_):
    JOB_IDX = FLAGS.job_idx
    # Read the list of samples to evaluate.
    mesh_path = "/fast/atatjer/finetuningrobustness/landscapes/stable_decay/pca_topk_meshgrid.pt"
    pca_path = "/fast/atatjer/finetuningrobustness/landscapes/stable_decay/pca_topk.pt"
    pca = torch.load(pca_path)
    mesh = torch.tensor(torch.load(mesh_path, weights_only=False)['meshgrid'])
    mean = pca['mean'].to('cuda')
    Vt = pca['topk_param_space'][:2, :].to('cuda')
    explained_variance_ratio = pca['explained_variance_ratio']
    sample = mesh[JOB_IDX].to('cuda')
    X_reconstructed = (sample @ Vt + mean).squeeze()

    print(X_reconstructed.shape)  # should be (N, P)
    # Reconstruct model architecture
    model_cfg = AutoConfig.from_pretrained(f"EleutherAI/pythia-160m")
    base_model =  AutoModelForCausalLM.from_config(model_cfg)
    breakpoint()
    def reconstruct_model_from_flat(base_model, flat_params):
        model = deepcopy(base_model)
        offset = 0
        with torch.no_grad():
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                numel = p.numel()
                shaped = flat_params[offset:offset+numel].view_as(p)
                p.copy_(shaped)
                offset += numel
        return model

    model = reconstruct_model_from_flat(base_model, X_reconstructed)
    print(model)
    breakpoint()
    # Evaluate
    # Requires many loss evaluations. Current val_set is  ~600kTKs, let's get an estimate for this, otherwise, it is not that terrible to have a run time of 2 minutes per each
    # considering that this can be parallelized for each of the models easily leading to less than 2k evaluations total, which is doable in the cluster.
    # this will serve as a template for an abstracted, config input, sweepable script.
    device = 'cuda'
    ds = load_from_disk('/fast/atatjer/hf_fast/datasets/refinedweb/')
    num_workers=8
    block_size=2048
    batch_size=20
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", trust_remote_code=True)
    def tokenize(example):
        example = tokenizer(example["content"], add_special_tokens=False)["input_ids"]
        return {"ids": example}
    tokenised = ds.map(
        tokenize,
        num_proc=num_workers,
        remove_columns=ds.column_names
        )
    concat_ids = [tok for ex in tokenised["ids"] for tok in ex]
    blocks = [concat_ids[i:i+block_size+1]           # +1 for the label shift
            for i in range(0, len(concat_ids) - (block_size+1), block_size)]

    dl = torch.utils.data.DataLoader(utils.BlockDataset(blocks),
                    batch_size=batch_size,
                    shuffle=False,
                    pin_memory=True)
    results = evaluation.eval(model, dl, device)
    # Log
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    os.environ["WANDB_SILENT"] = "true"
    wandb.init(
        project="loss_landscape",
        name=f"{JOB_IDX}",
        dir="/fast/atatjer/wandb/scalinglawsquantization",
        config={
            "JOB_IDX":JOB_IDX,
            "sample": sample,
            },
    )
    results = {
        **results,
        "sample": sample.tolist(),
        "samples_path":samples_path,
        "pca_path":pca_path,
        }
    logging.info(f"Results {results}")
    wandb.log(results)
    
    
if __name__ == "__main__":
    app.run(main)
