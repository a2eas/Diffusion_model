import torch



def Noise_Scheduler(t = 1000,beta_start =1e-4,beta_end = 0.02,device='cpu'):
    beats = torch.linspace(beta_start,beta_end,t,device=device)
    alphas = 1.0 - beats
    alphas_bar = torch.cumprod(alphas,dim=0)

    return beats,alphas,alphas_bar