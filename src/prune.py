import re

import torch.nn.utils.prune as prune

pruning_map = {
    'l2_structured': prune.ln_structured
}

def do_prune(model, pruning_method, pruning_rate):
    # Check method exist
    if pruning_method not in pruning_map.keys():
        raise ValueError(f"{pruning_method} not in pruning_map")
    # Get pruning
    pruning = pruning_map['pruning_method']
    # L_n Norm Structured
    p = re.compile('l\d+_structured')
    if p.match(pruning_method) is not None:
        model = pruning(model, 'weight', amount=pruning_rate, dim=-1)
    return model