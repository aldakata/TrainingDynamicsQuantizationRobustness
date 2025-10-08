"""
    Quantize model given config file.
    Push to HF_HUB with consistent revision name.
"""
from filelock import SoftFileLock
import filelock
# Override FileLock globally to use SoftFileLock
filelock.FileLock = SoftFileLock

from absl import app, flags, logging
from collections import defaultdict

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.utils.eval import EVAL

from datasets import load_dataset, load_from_disk
import torch, math
import wandb

import utils
import evaluation
import importer as importer

import os
import shutil
from tqdm import tqdm

from vllm import LLM, SamplingParams
from lm_eval.models.vllm_causallms import VLLM
from lm_eval import simple_evaluate
import torch.distributed as dist
import atexit

def cleanup_process_group():
    """Clean up distributed process group on exit"""
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except:
            pass

atexit.register(cleanup_process_group)

flags.DEFINE_string('config', 'config/config.yaml', 'Path to config.yaml file.')
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
FLAGS = flags.FLAGS

def extract_accuracies(results):
    """Extract just accuracy scores from lm-eval results"""
    main_results = results['results']
    accuracies = {}
    for task_name, task_results in main_results.items():
        if 'acc,none' in task_results:
            accuracies[task_name] = task_results['acc,none']
        elif 'acc_norm,none' in task_results:
            accuracies[task_name] = task_results['acc_norm,none']
    return accuracies


def _quantize(cfg, saved_quantized):
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
        )
    modelQ.save(saved_quantized)
    del modelQ
    torch.cuda.empty_cache()
    logging.info(f"Saved quantized model to {saved_quantized}")

def main(_):
    CFG_PATH, JOB_IDX = FLAGS.config, FLAGS.job_idx
    logging.info(f"CFG: {CFG_PATH} JOB_IDX: {JOB_IDX}")
    cfg, _ = utils.load_config(CFG_PATH, JOB_IDX)
    logging.info(cfg)
    if cfg.debug:
        import pdb; pdb.set_trace()

    os.environ["WANDB__SERVICE_WAIT"] = "600"
    os.environ["WANDB_SILENT"] = "true"
    wandb.init(
        project='trainingdynamicstaskeval',
        name=cfg.wandb_run_name,
        dir=cfg.wandb_dir,
        config=cfg._asdict(),
    )
    
    logging.info(f"Working with {cfg.repo_origin} revision: {cfg.revision}")
    quantization_backend = utils.get_quantization_backend(cfg)
    logging.info(f"Quantization backend: {quantization_backend}")
    
    assert quantization_backend.lower()=='gptq', f"quantization_backend={quantization_backend} not accepted."
    revision = f"{cfg.repo_origin}+{cfg.revision}".replace('/','_')

    saved_quantized = importer.get_quantized_path(
        quantized_dir=cfg.model_quantized_dir,
        repo_origin=cfg.repo_origin,
        revision=cfg.revision,
        q_bits=cfg.q_bits,
        suffix="_HF",
        quantization_backend=quantization_backend,
    )
    model_path = cfg.repo_origin
    kwargs = {}
    if cfg.quantize:
        if not os.path.exists(saved_quantized):
            logging.info(f"Quantized model not found at {saved_quantized}. Will quantize.")
            _quantize(cfg, saved_quantized)
            logging.info(f"Loaded quantized model from {saved_quantized}")
        elif 'smol' in cfg.repo_origin.lower():
            logging.info(f"Quantized model found at {saved_quantized}.")
        else:
            logging.info(f"Quantized model found at {saved_quantized}. Will load.")
            try:
                modelQ = GPTQModel.load(
                    saved_quantized,
                    device_map="auto",
                    trust_remote_code=True,
                    backend='torch',
                    )
            except Exception as e:
                logging.warning(f"Could not load the model at {saved_quantized} because of error {e}. Will delete and re-quantize.")
                shutil.rmtree(saved_quantized)
                _quantize(cfg, saved_quantized)
        model_path = saved_quantized
    else:
        model_path = cfg.repo_origin
        kwargs = {'revision': cfg.revision}
        saved_quantized = f'{cfg.repo_origin}/{cfg.revision}'
    
    llm = VLLM(
        pretrained=model_path,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=0.80,
        batch_size="auto",
        dtype='float16',
        data_parallel_size=1,
        tensor_parallel_size=1,
        **kwargs
        )

    tasks = [
        "arc_challenge",  
        "arc_easy",       
        "openbookqa",     
        "piqa",           
        "hellaswag",      
        "winogrande",     
        "mathqa",         
        "pubmedqa",       
        "sciq",           
        "social_iqa",     
        "commonsense_qa", 
        "mmlu"            
    ]
    results = simple_evaluate(
        model=llm,
        tasks=tasks,
        limit=None,
        batch_size="auto",
        num_fewshot=5,
    )
    results = extract_accuracies(results)
    results = {
        **results,
        'quantization_backend':quantization_backend,
        'saved_path': saved_quantized,
        }
    logging.info(f"quantized {results}")
    wandb.log(results)
    cleanup_process_group()

if __name__ == "__main__":
    app.run(main)