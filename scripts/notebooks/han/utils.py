import numpy as np
import torch
from tqdm.auto import tqdm


def get_score_from_embs(embs, model, batch_size, preprocess=None):
    
    scores = list()

    with torch.no_grad():

        for i in tqdm(range(0, len(embs), batch_size), leave=False):
            
            x = torch.tensor(embs[i:i+batch_size]).cuda()
            
            if preprocess is not None:
                x = preprocess(x)

            score = model(x)[..., 0]

            scores.append(score.detach().cpu().numpy())

    scores = np.concatenate(scores, axis=0)
    
    return scores

def init_weights(model):

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(
                m.weight, mode='fan_in', nonlinearity='linear')
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            
            
def log_likelihood(mu, sigma2, targets, eps=1e-5, clip=None):
    
    loss = (((targets - mu) ** 2) / (sigma2 + eps) + torch.log(sigma2 + eps)) / 2
    
    loss = torch.nan_to_num(loss, 0)
    
    if clip is not None:
        loss = torch.clip(loss, -clip, clip)
    
    return loss.mean()

def cross_entropy_likelihood(mu, sigma2, targets, dim=-1, eps=1e-5, clip=None):
    
    ex = torch.exp(mu + sigma2 / 2)
    
    loss = - (mu.gather(dim, targets.unsqueeze(dim)).squeeze(dim) - torch.log(ex.sum(dim=dim) + eps))
    
    loss = torch.nan_to_num(loss, 0)
    
    if clip is not None:
        loss = torch.clip(loss, -clip, clip)
    
    return loss.mean()


def pearson_correlation(u, v=None, eps=1e-6):
    
    u = u - torch.mean(u, dim=-1).unsqueeze(-1)
    u_ = torch.sqrt(torch.sum(u ** 2, dim=-1))
    
    if v is not None:
        v = v - torch.mean(v, dim=-1).unsqueeze(-1)
        v_ = torch.sqrt(torch.sum(v ** 2, dim=-1))
    else:
        v = u
        v_ = u_

    return torch.sum(u.unsqueeze(-2) * v.unsqueeze(-3), dim=-1) / (u_.unsqueeze(-1) * v_.unsqueeze(-2) + eps)

