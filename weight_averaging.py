from filelock import SoftFileLock
import filelock
# Override FileLock globally to use SoftFileLock
filelock.FileLock = SoftFileLock


from transformers import AutoTokenizer, GPTQConfig, AwqConfig, AutoConfig, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
import torch, math
import wandb
from gptqmodel import GPTQModel, QuantizeConfig

from huggingface_hub import list_repo_refs

import utils
import evaluation
import importer

import os
from absl import app, flags, logging

flags.DEFINE_string('config', 'config/config.yaml', 'Path to config.yaml file.')
flags.DEFINE_integer('job_idx', None, 'Job idx for job-array sweeps. From 0 to n-1.')
FLAGS = flags.FLAGS

def get_model_cfg(repo_origin):
    repo_origin = repo_origin.lower()
    if ('pythia' in repo_origin) or ('olmo' in repo_origin):
        return repo_origin
    elif 'smol' in repo_origin:
        return 'HuggingFaceTB/SmolLM3-3B'
    else:
        raise ValueError(f"repo_origin {repo_origin} not recognized.")

def main(_):
    """
        1. Pass repo_origin and revision via config.
        3. Sample with the corresponding parameter L. 
        4. Convert models to torch.
        5. Average model.
        6. Quantize with GPTQ.
        7. Save quantized and Averaged
        8. Evaluate both.
    """
    CFG_PATH, JOB_IDX = FLAGS.config, FLAGS.job_idx
    print(f"CFG: {CFG_PATH} JOB_IDX: {JOB_IDX}")
    cfg, _ = utils.load_config(CFG_PATH, JOB_IDX)
    print(cfg)
    utils.init_wandb(cfg)
    
    repo_origin = cfg.repo_origin
    tokenizer_name = cfg.tokenizer_name
    q_bits = cfg.q_bits
    block_name_to_quantize = cfg.block_name_to_quantize
    dataset_path = cfg.dataset_path
    quantization_backend = utils.get_quantization_backend(cfg)
    batch_size =  cfg.batch_size
    num_workers = cfg.num_workers
    block_size =  cfg.block_size
    device = cfg.device
    
    cfg_model = get_model_cfg(repo_origin)
    L = cfg.L
    LAWA_DST = os.environ.get('_CONDOR_SCRATCH_DIR')+'/tmp/'

    revisions = cfg.revision
    models = []
    names = []
    print(f"Working with {cfg.repo_origin} L: {cfg.L}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True) 
    for i in range(L):
        j=i+1
        revision = revisions[-j]
        model = importer.from_pretrained(
                    repo_origin,
                    revision=revision,
                    device_map='cpu',
                    low_cpu_mem_usage=True,
                ) 
        models.append(model)
        names.append(revision)

    model_cfg = AutoConfig.from_pretrained(cfg_model)
    with torch.no_grad():
        for name, layer in model.state_dict().items():
            print(f"Averaging layer: {name}")
            tmp = torch.mean(torch.stack([m.state_dict()[name].data for m in models]), dim=0)
            model.state_dict()[name].data.copy_(tmp)
    print("Model averaging finished")

    dst = f"lawa_{repo_origin}_{L}_{cfg.q_bits}"
    dst = os.path.join(LAWA_DST, dst) 
    model.cpu()
    model.save_pretrained(dst, safe_serialization=True) # Important for quantization
    tokenizer.save_pretrained(dst)
    print(f"LAWA model saved at {dst}.")

    model = model.to(device)    
    
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
        dst, 
        quant_config, 
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
    quantized_path=f"{dst}_{cfg.q_bits}bits"
    modelQ.save(quantized_path)
    logging.info(f"Saved quantized model to {quantized_path}")

    modelQ = GPTQModel.load(
        quantized_path, 
        quant_config, 
        trust_remote_code=True,
        torch_dtype='auto',
        device_map="auto",
        )

    saved_quantized = f"{dst}_{quantization_backend}"
    if os.path.isdir(dataset_path):
        ds = load_from_disk(dataset_path)
    else:
        ds = load_dataset(dataset_path, dataset_name, split='validation')

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

    print("Evaluation")
    results = evaluation.eval_battle(modelQ, model, dl, device)
    results = {
        **results,
        'quantization_backend':quantization_backend,
        'eval_dataset': dataset_path,
        'saved_path': saved_quantized,
        }
    print(f"quantized {results}")
    wandb.log(results)

    
if __name__=='__main__':
    app.run(main)