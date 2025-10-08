import os
import yaml
import wandb
import torch

from itertools import product
from collections import namedtuple

def load_config(path, job_idx=None):
    """
    Parse a yaml file and return the correspondent config as a namedtuple.
    If the config files has multiple entries, returns the one corresponding to job_idx.
    """
    with open(path, "r") as file:
        config_dict = yaml.safe_load(file)
    Config = namedtuple("Config", config_dict.keys())

    if job_idx is None:
        cfg = config_dict
        sweep_size = 1

    else:
        keys = list(config_dict.keys())
        values = [
            val if isinstance(val, list) else [val] for val in config_dict.values()
        ]
        combinations = list(product(*values))

        sweep_size = len(combinations)
        if job_idx >= sweep_size:
            raise ValueError(
                "job_idx exceeds the total number of hyperparam combinations."
            )

        combination = combinations[job_idx]
        cfg = {keys[i]: combination[i] for i in range(len(keys))}

    return Config(**cfg), sweep_size

def init_wandb(cfg):
    if cfg.debug:
        return

    """Initalizes a wandb run"""
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    os.environ["WANDB_SILENT"] = "true"
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        dir=cfg.wandb_dir,
        config=cfg._asdict(),
    )

def get_model_size_gb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    return total_size / (1024 ** 3)

def get_quantization_backend(cfg):
    i = 0
    name = 'baseline'
    if cfg.gptq:
        name = "GPTQ"
        i += 1
    if cfg.awq:
        name = "AWQ"
        i += 1
    if cfg.bnb:
        name = "BNB"
        i += 1
    if i > 1:
        raise ValueError(f"{i} quantization backends selected. Please select one.")
    return name

class BlockDataset(torch.utils.data.Dataset):
    def __init__(self, blocks):
        self.blocks = blocks
    def __len__(self):
        return len(self.blocks)
    def __getitem__(self, idx):
        x = self.blocks[idx]
        return {
            "input_ids": torch.tensor(x[:-1], dtype=torch.long),
            "labels":    torch.tensor(x[1:],  dtype=torch.long)
        }
