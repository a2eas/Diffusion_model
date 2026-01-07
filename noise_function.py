import cv2
import torch
import math
from noise_scheduler import Noise_Scheduler
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else 'cpu' # setting the device to gpu if avaliable 
def reverse_noise(noise_image,t,eps_hat,alphas,alphas_hat):
    #t = timesteps 
    #eps nosise added
    #alpahs how much of the image is perseved at timestep t
    sqrt_alpha = torch.sqrt(alphas) # squarte root of alphas at time t
    sqrt_alpha_minus_one = torch.sqrt(alphas_hat) # minus one also square root
    x_hat = (noise_image- sqrt_alpha_minus_one * eps_hat) / sqrt_alpha # formual for getting the og image before noise
    return x_hat
def noise_foward(image,t,alphas,alphas_hat):
    image = torch.tensor(image)
    image = image.float() /255.0
    eps = torch.randn_like(image) #genrating random noise in the same size as the image
    sqrt_alpahs = torch.sqrt(alphas_hat[t])
    sqrt_alpahs_hat = torch.sqrt(1.0 - alphas_hat[t])
    x_t = sqrt_alpahs*image + sqrt_alpahs_hat*eps # formual for adding noise to the image
    return x_t,eps

if __name__ == "__main__":
    img = cv2.imread(r'Screenshot 2025-12-24 223016.png')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    b,a,a_bar = Noise_Scheduler(t=1000)
    x_t , eps = noise_foward(img,401,a,a_bar)
    print('alphas_bar[t]', a_bar[999].item())

    plt.imshow(x_t.clamp(-2,2), cmap="gray")
    plt.axis("off")
    plt.show()

