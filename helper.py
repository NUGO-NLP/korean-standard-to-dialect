import time
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs    

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            # nn.init.normal_(param.data, mean=0, std=0.01)
            nn.init.xavier_normal_(param.data)
        else:
            nn.init.constant_(param.data, 0)

    # for p in model.parameters():
    #     if p.dim() > 1:
    #         nn.init.xavier_uniform(p)
