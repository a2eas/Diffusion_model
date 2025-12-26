import cv2
import torch
import math


device = "cuda" if torch.cuda.is_available() else 'cpu'
def reverse_noise(og_image,t,eps_hat,alphas):
    sqrt_alpha = torch.sqrt(alphas[t])
    sqrt_alpha_minus_one = torch.sqrt(1-alphas[t])
    x_hat = (og_image- sqrt_alpha_minus_one * eps_hat) / sqrt_alpha
    return x_hat
def noise_foward(image,t,alphas):
    eps = torch.randn_like(image)
    sqrt_alpahs = torch.sqrt(alphas[t])
    sqrt_alpahs_hat = torch.sqrt(1-alphas[t])
    x_t = sqrt_alpahs*image + sqrt_alpahs_hat*eps
    return x_t,eps