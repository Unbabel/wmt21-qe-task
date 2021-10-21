import torch
import torch.nn as nn

class VarianceLoss(nn.Module):
    
    def forward(self, mu: torch.Tensor, std: torch.Tensor, target: torch.Tensor):
        sigma = std**2
        log1 = 0.5 * torch.neg(torch.log(sigma)).exp() 
        mse = (target - mu)**2
        log2 = 0.5 * torch.log(sigma)
        return torch.sum(log1*mse+log2)
