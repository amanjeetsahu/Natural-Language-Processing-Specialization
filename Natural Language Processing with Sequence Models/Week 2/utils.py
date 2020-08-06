import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_batch(source, i):
    '''
        returns a batch
    '''
    bptt = 35
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    
    return data, target


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


# to detach the hidden state from the graph.
def detach(hidden):
    """
    This function detaches every single tensor. 
    """
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(detach(v) for v in hidden)