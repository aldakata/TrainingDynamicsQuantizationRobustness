"""
    Script to download a bunch of model revision to the $HF_HOME/hub
    Dirty/interactive to read all the checkpoints and then paste exactly the ones
    needed. This should be done only once for each model, and you probably do not want
    evenly spaced checkpoints, and each model family has its own perks e.g. ingredients and such
    that deserve their own manual attention.
"""

from filelock import SoftFileLock
import filelock
# Override FileLock globally to use SoftFileLock
filelock.FileLock = SoftFileLock

from absl import app, flags, logging
from collections import defaultdict

from huggingface_hub import list_repo_refs

from transformers import AutoTokenizer, GPTQConfig, AwqConfig
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi, snapshot_download

import torch, math
import numpy as np
import wandb

import utils
import evaluation
import importer

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" # RUST!
handy_list = [
        'swiss-ai/Apertus-8B-2509',
        'open-sci/open-sci-ref-v0.01-1.7b-nemotron-hq-1T-4096',
        'allenai/OLMo-2-0325-32B',
        'allenai/OLMo-2-0325-32B',
        'allenai/OLMo-2-1124-7B',
        'allenai/OLMo-2-1124-13B',
        'allenai/OLMo-2-0425-1B',
        'LLM360/Amber',
        ]
flags.DEFINE_string('repo_origin', 'allenai/OLMo-1B', 'repo_origin')
flags.DEFINE_integer('period', 10, 'period of sampling revision')
FLAGS = flags.FLAGS

def main(_):
    out = list_repo_refs(FLAGS.repo_origin)
    all_revisions = [b.name for b in out.branches]
    if False:
        print(f"Found revisions {len(all_revisions)}: [\n        '" +"',\n        '".join(all_revisions)+"',\n    ]")
        exit()
    all_revisions =     [
        "step636000-tokens2668B",
        "step558000-tokens2340B",
        "step516250-tokens2165B",
        "step437000-tokens1833B",
        "step388000-tokens1627B",
        "step114000-tokens478B",
        "step63000-tokens264B",
        "step40000-tokens168B",
        "step330000-tokens1384B",
        "step117850-tokens494B",
        "step90000-tokens377B",
        "step738020-tokens3095B",
    ]


    if not all_revisions:
        raise ValueError(f"No revisions found for {FLAGS.repo_origin}. Please check the model ID.")
    print(f"Found {len(all_revisions)} unique revisions to download model files from.")
    print("Proceeding to download...")
    repo_origin = FLAGS.repo_origin

    for i, revision_hash in enumerate(list(all_revisions)): # Sort for consistent order
        try:
            print(f"\n[{i+1}/{len(all_revisions)}] Downloading model files for revision: {revision_hash}")
            download_info = snapshot_download(
                repo_id=FLAGS.repo_origin,
                revision=revision_hash,
            )
            print(f"Downloaded model files for {FLAGS.repo_origin} revision {revision_hash}")

        except Exception as e:
            print(f"Error downloading revision {FLAGS.repo_origin} {revision_hash}: {e}")

    print(f"\nDownload complete for model files of all listed revisions to {os.environ.get('HF_HOME')}.")
    print("You can now find each revision's model files in their respective subdirectories.")

if __name__ == "__main__":
    app.run(main)
