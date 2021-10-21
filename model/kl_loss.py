import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

    
    
class KLLoss(nn.Module):

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, target_mu: torch.Tensor, target_std: torch.Tensor):   
        # Add fudge factor to variance to avoid large KL values
        #   (value of 1e-2 just turned out to work - 1e-3 already
        #   occasionally caused loss > 1000)
        std1 = target_std
        std2 = sigma
        mean1 = target_mu
        mean2 = mu
        kl = torch.log(torch.abs(std2)/torch.abs(std1)) + (std1**2 + (mean1 - mean2)**2)/(2*std2**2) - 0.5
        
        return kl.mean()
