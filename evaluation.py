"""
    Eval loops
"""
import torch, math
from tqdm import tqdm

def calculate_linear_grad_norm(model, device):
    total_norm = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if isinstance(model.get_submodule(name.rsplit('.', 1)[0]), torch.nn.Linear):
            if param.grad is not None:
                total_norm += param.grad.detach().norm(2)
    for param in model.parameters():
        param.grad = None
    return total_norm

def eval(model, dataloader, device):
    total_nll, total_tokens = 0.0, 0
    model.to(device)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            outputs = model(batch['input_ids'].to(device), labels=batch['input_ids'].to(device))
            loss = outputs.loss
            n_tokens = batch["labels"].numel()
            total_nll    += loss.item() * n_tokens
            total_tokens += n_tokens
        cross_entropy = total_nll / total_tokens
        ppl           = math.exp(cross_entropy)

    return {
        "cross_entropy":cross_entropy, 
        "ppl": ppl,
        }

def eval_battle(modelQ, modelB, dataloader, device):
    """
        !The gradient computation triggers CUDA OOM for >6B models.
    """
    total_nllQ, total_nllB, total_tokens = 0.0, 0.0, 0
    gradient_normB, gradient_normQ = [], []
    modelQ.eval()
    modelB.eval()
    for batch in tqdm(dataloader):
        outputsB = modelB(batch['input_ids'].to(device), labels=batch['input_ids'].to(device))
        lossB = outputsB.loss
        lossB.backward() 
        gradient_normB.append(calculate_linear_grad_norm(modelB, device))
        n_tokens = batch["labels"].numel()
        total_nllB    += lossB.item() * n_tokens
        outputsQ = modelQ(batch['input_ids'].to(device), labels=batch['input_ids'].to(device))
        lossQ = outputsQ.loss
        total_nllQ    += lossQ.item() * n_tokens 
        total_tokens += n_tokens

    cross_entropyQ = total_nllQ / total_tokens        
    pplQ           = math.exp(cross_entropyQ)
    cross_entropyB = total_nllB / total_tokens
    pplB           = math.exp(cross_entropyB)

    params_B = torch.cat([p.view(-1) for p in modelB.parameters()])
    params_Q = torch.cat([p.view(-1) for p in modelQ.parameters()])

    weight_l1_B = params_B.norm(p=1)
    weight_l1_Q = params_Q.norm(p=1)

    weight_l2_B = params_B.norm(p=2)
    weight_l2_Q = params_Q.norm(p=2)

    weight_linf_B = params_B.norm(p=float('inf'))
    weight_linf_Q = params_Q.norm(p=float('inf'))

    gradient_normB = (sum(gradient_normB)/total_tokens).item()
    gradient_normQ = sum(gradient_normQ)/total_tokens
    return {
        "cross_entropyQ":cross_entropyQ, 
        "pplQ": pplQ,
        "cross_entropyB":cross_entropyB, 
        "pplB": pplB,
        "delta_cross_entropy":cross_entropyQ-cross_entropyB,
        "delta_ppl":pplQ-pplB,
        "gradient_normB": gradient_normB, 
        "gradient_normQ": gradient_normQ, 
        "weight_l1_B": weight_l1_B.item(),
        "weight_l1_Q": weight_l1_Q.item(),
        "weight_l2_B": weight_l2_B.item(),
        "weight_l2_Q": weight_l2_Q.item(),
        "weight_linf_B": weight_linf_B.item(),
        "weight_linf_Q": weight_linf_Q.item(),
        }
