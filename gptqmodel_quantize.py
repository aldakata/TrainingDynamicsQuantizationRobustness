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

from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig

from datasets import load_dataset, load_from_disk
import torch, math
import wandb

import utils
import evaluation
import importer as importer

import os

flags.DEFINE_string('config', 'config/config.yaml', 'Path to config.yaml file.')
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
FLAGS = flags.FLAGS

def _eval(model, cfg):
    ds = load_from_disk(cfg.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, trust_remote_code=True)

    logging.info("Evaluation")
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
    results = evaluation.eval(model, dl, cfg.device)
    
    del model
    del dl
    torch.cuda.empty_cache()
    return results

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
    
    assert quantization_backend.lower()=='gptq', f"quantization_backend={quantization_backend} not accepted."
    revision = f"{cfg.repo_origin}+{cfg.revision}".replace('/','_')

    # get the name of the quantized model
    saved_quantized = importer.get_quantized_path(
        quantized_dir=cfg.model_quantized_dir,
        repo_origin=cfg.repo_origin,
        revision=cfg.revision,
        q_bits=cfg.q_bits,
        suffix="_HF",
        quantization_backend=quantization_backend,
    ) # Needed for the result DF
    should_load = os.path.isdir(saved_quantized) and cfg.use_quant_cache
    if should_load:
        try:
            modelQ = GPTQModel.load(
                saved_quantized,
                device_map="auto",
                trust_remote_code=True,
                backend='torch',
                )
            logging.info(f"Loaded quantized model from {saved_quantized}")
        except Exception as e:
            logging.warning(f"Could not load the model at {saved_quantized} because of error {e}. Will re-quantize.")
            should_load = False
    if not should_load:
        calibration_dataset = load_dataset(
            "allenai/c4",
            data_files="en/c4-train.00001-of-01024.json.gz",
            split="train"
        ).select(range(1024))["text"]

        quant_config = QuantizeConfig(
            bits=cfg.q_bits,
            group_size=cfg.group_size,
        )
        modelQ = GPTQModel.load(
            cfg.repo_origin, 
            quant_config, 
            revision=cfg.revision,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map="auto",
            )

        logging.info(f"Quantizing {cfg.repo_origin}+{cfg.revision} to {cfg.q_bits} bits with group_size {cfg.group_size}")
        modelQ.quantize(
                calibration_dataset, 
                batch_size=cfg.calibration_batch_size,
                auto_gc = False,
                backend="torch",
                # backend = 'triton',
                # backend = 'marlin',
            )
        modelQ.to(cfg.device)
        
        modelQ.save(saved_quantized)
        logging.info(f"Saved quantized model to {saved_quantized}")
        # GPTQModel does not support inference after quant. it has to be saved and loaded again.
        modelQ = GPTQModel.load(
                saved_quantized,
                device_map="auto",
                trust_remote_code=True,
                backend='torch',
                )
        logging.info(f"Loaded quantized model from {saved_quantized}")
    
    resultsQ = _eval(modelQ, cfg)
    logging.info(f"Quantized results \n{resultsQ}")
    
    # Base model
    modelB = importer.from_pretrained(
            cfg.repo_origin,
            revision=cfg.revision,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
    )
    resultsB = _eval(modelB, cfg)
    logging.info(f"Base results \n{resultsB}")
    results = {
        "cross_entropyQ":resultsQ['cross_entropy'], 
        "cross_entropyB":resultsB['cross_entropy'], 
        "pplQ": resultsQ['ppl'], 
        "pplB": resultsB['ppl'],
        "delta_cross_entropy":resultsQ['cross_entropy']-resultsB['cross_entropy'],
        "delta_ppl":resultsQ['ppl']-resultsB['ppl'],
    }
    results = {
        **results,
        'quantization_backend':quantization_backend,
        'saved_path': saved_quantized,
        }
    logging.info(f"quantized {results}")
    wandb.log(results)

if __name__ == "__main__":
    app.run(main)