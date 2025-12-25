import cv2
device = "cuda" if torch.cuda.is_avaliable() else 'cpu'
def noise(og_image,schedule_type,time_step,epslion,T,noise_mean=0,noise_variance=1):
    og_image = cv2.resize(og_image,(64,64)) /255.0
    og_image = cv2.cvtColor(og_image,cv2.COLOR_BGR2RGB)