import cv2
import torch
import math


device = "cuda" if torch.cuda.is_available() else 'cpu' # setting the device to gpu if avaliable 
def reverse_noise(og_image,t,eps_hat,alphas):
    #t = timesteps 
    #eps nosise added
    #alpahs how much of the image is perseved at timestep t
    sqrt_alpha = torch.sqrt(alphas[t]) # squarte root of alphas at time t
    sqrt_alpha_minus_one = torch.sqrt(1-alphas[t]) # minus one also square root
    x_hat = (og_image- sqrt_alpha_minus_one * eps_hat) / sqrt_alpha # formual for getting the og image before noise
    return x_hat
def noise_foward(image,t,alphas):
    eps = torch.randn_like(image) #genrating random noise in the same size as the image
    sqrt_alpahs = torch.sqrt(alphas[t])
    sqrt_alpahs_hat = torch.sqrt(1-alphas[t])
    x_t = sqrt_alpahs*image + sqrt_alpahs_hat*eps # formual for adding noise to the image
    return x_t,eps