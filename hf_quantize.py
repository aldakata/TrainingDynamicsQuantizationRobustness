"""
    Quantize modelQ given config file.
    Push to HF_HUB with consistent revision name.
"""
from filelock import SoftFileLock
import filelock
# Override FileLock globally to use SoftFileLock
filelock.FileLock = SoftFileLock

from absl import app, flags, logging
from collections import defaultdict

from transformers import AutoTokenizer, GPTQConfig, AwqConfig
from datasets import load_dataset, load_from_disk
import torch, math
import wandb

import utils
import evaluation
import importer

import os

flags.DEFINE_string('config', 'config/config.yaml', 'Path to config.yaml file.')
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
FLAGS = flags.FLAGS

def main(_):
    CFG_PATH, JOB_IDX = FLAGS.config, FLAGS.job_idx
    logging.info(f"CFG: {CFG_PATH} JOB_IDX: {JOB_IDX}")
    cfg, _ = utils.load_config(CFG_PATH, JOB_IDX)
    logging.info(cfg)
    if cfg.debug:
        import pdb; pdb.set_trace()

    utils.init_wandb(cfg)
    logging.info(f"Working with {cfg.repo_origin} revision: {cfg.revision}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, trust_remote_code=True)
    quantization_backend = utils.get_quantization_backend(cfg)
    logging.info(f"Quantization backend: {quantization_backend}")
    if quantization_backend.lower()=='gptq':
        q_config = GPTQConfig(
            bits=cfg.q_bits, 
            block_name_to_quantize=cfg.block_name_to_quantize,
            dataset=cfg.gptq_dataset,
            tokenizer=tokenizer,
            model_seqlen=cfg.gptq_model_seqlength,
            # backend='triton', # Doesn't support 3bit quant.
            backend='torch',
            )
    elif quantization_backend.lower()=='bnb':
        from transformers import BitsAndBytesConfig
        if cfg.q_bits==8:
            q_config = BitsAndBytesConfig(load_in_8bit=True)
        elif cfg.q_bits==4:
            q_config = BitsAndBytesConfig(load_in_4bit=True)        
        else:
            raise ValueError("BitsAndBytesConfig only supports 4- or 8-bit quantization.")
    else:
        raise ValueError(f"quantization_backend={quantization_backend} not accepted.")

    modelQ = importer.from_pretrained( 
            cfg.repo_origin, 
            revision=cfg.revision,
            device_map="auto", 
            quantization_config=q_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    saved_quantized = importer.get_quantized_path(
            quantized_dir=cfg.model_quantized_dir,
            repo_origin=cfg.repo_origin,
            revision=cfg.revision,
            q_bits=cfg.q_bits,
            suffix="_HF",
            quantization_backend=quantization_backend,
        )

    modelB = importer.from_pretrained(
            cfg.repo_origin,
            revision=cfg.revision,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    if os.path.isdir(cfg.dataset_path):
        ds = load_from_disk(cfg.dataset_path)
    else:
        ds = load_dataset(cfg.dataset_path, cfg.dataset_name, split='validation')

    def tokenize(example):
        example = tokenizer(example["content"], add_special_tokens=False)["input_ids"]
        return {"ids": example}

    tokenised = ds.map(
        tokenize,
        num_proc=cfg.num_workers,
        remove_columns=ds.column_names
        )

    concat_ids = [tok for ex in tokenised["ids"] for tok in ex]
    blocks = [concat_ids[i:i+cfg.block_size+1]           # +1 for the label shift
            for i in range(0, len(concat_ids) - (cfg.block_size+1), cfg.block_size)]

    dl = torch.utils.data.DataLoader(utils.BlockDataset(blocks),
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    pin_memory=True)

    logging.info("Evaluation")
    results = evaluation.eval_battle(modelQ, modelB, dl, cfg.device)
    results = {
        **results,
        'quantization_backend': quantization_backend,
        'saved_path': saved_quantized,
        }
    logging.info(f"quantized {results}")
    wandb.log(results)

if __name__ == "__main__":
    app.run(main)
