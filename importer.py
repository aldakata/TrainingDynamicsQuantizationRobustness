import os
import logging
import utils
logger = logging.getLogger(__name__)

def get_quantized_path(
        quantized_dir, 
        repo_origin,
        revision,
        q_bits, 
        suffix,
        quantization_backend
    ):
    return f"{quantized_dir}/{repo_origin.replace('/', '_')}/{revision}_{q_bits}bits_{quantization_backend}{suffix}/"

def from_pretrained(repo_origin, **kwargs):
    if(
        "smol" in repo_origin.lower()
        or "olmo-2-" in repo_origin.lower()
        or "dots.llm1" in repo_origin.lower()
        or 'swiss-ai' in repo_origin.lower()
        or 'open-sci' in repo_origin.lower()
        or 'amber' in repo_origin.lower()
    ):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(repo_origin, **kwargs)
    elif "olmo" in repo_origin.lower():
        from hf_olmo import OLMoForCausalLM  # pip install ai2-olmo
        return OLMoForCausalLM.from_pretrained(repo_origin, **kwargs)
    elif "pythia" in repo_origin.lower():
        from transformers import GPTNeoXForCausalLM, AutoTokenizer
        return GPTNeoXForCausalLM.from_pretrained(repo_origin, **kwargs)
    else:
        raise NotImplementedError(f"{repo_origin} not implemented yet")