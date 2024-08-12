from typing import List
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd

def calculate_log_prob(texts: List[str], 
                       tokenizer: AutoTokenizer, 
                       model: AutoModelForCausalLM, 
                       device: torch.device):
    """
    Calculate the log probabilities of a batch of texts using the language model.
    """
    encodings = tokenizer(texts, 
                          return_tensors='pt', 
                          truncation=True,
                          max_length=512, 
                          padding=True)
    
    encodings["input_ids"] = encodings["input_ids"].to(device)
    encodings["attention_mask"] = encodings["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits

    logits = logits[:, :-1, :].contiguous()
    probs = logits - logits.logsumexp(2, keepdim=True)
    labels = encodings["input_ids"][:, 1:].contiguous()
    attention_mask = encodings["attention_mask"][:, 1:].contiguous()
    
    target_probs = torch.gather(probs, 2, labels.view(probs.shape[0], probs.shape[1], 1))
    target_probs = target_probs.view(target_probs.shape[0], target_probs.shape[1])
    target_probs = target_probs*attention_mask 

    sum_loss = target_probs.sum(dim=1)
    token_count = attention_mask.sum(dim=1)

    log_probs = sum_loss/token_count
    return log_probs

def calculate_perplexity(texts: List[str], 
                         tokenizer: AutoTokenizer, 
                         model: AutoModelForCausalLM, 
                         device: torch.device):
    """
    Calculate the perplexity of a batch of texts using the language model.
    """
    log_probs = calculate_log_prob(texts, tokenizer, model, device)
    perplexities = torch.exp(-log_probs)
    return perplexities.tolist()

def calculate_ucb(h: dict, n: dict, N: int, ucb_c: float) -> dict:
    """
    Calculate the Upper Confidence Bound (UCB) for each trigger.

    Args:
        h (dict): Dictionary containing the average historical scores of each trigger.
        n (dict): Dictionary containing the number of times each trigger has been sampled.
        N (int): Total number of samples across all triggers.
        ucb_c (float): Coefficient for the confidence interval, controlling exploration.

    Returns:
        dict: Dictionary of triggers with their UCB values.
    """
    ucb = {}
    for trigger in h:
        count = n.get(trigger, 0) + 1
        ucb_value = h[trigger] + ucb_c * np.sqrt(np.log(N) / count)
        ucb[trigger] = ucb_value

    return ucb