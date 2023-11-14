import torch
import math
from tqdm import tqdm


class Diffusion:
    def __init__(self, noise_step=1000, img_size=64, device='mps') -> None:
        self.noise_step = noise_step
        self.img_size = img_size
        self.device = device
        # this alpha_hat is the alpha_hat for all noise step
        self.beta, self.alpha, self.alpha_hat = self.prepared_noise_schedule(noise_step)
        self.alpha_hat = self.alpha_hat.to(self.device)
        self.beta = self.beta.to(self.device)
        self.alpha = self.alpha.to(self.device)
    
    def prepared_noise_schedule(self, noise_step=1000, offset=0.008) -> (torch.tensor, torch.tensor, torch.tensor):
        '''
            per openai paper https://arxiv.org/pdf/2102.09672.pdf
            beta [Vector]        : beta governs the guassian distribution.
                        originally is a vector between a lower and upper bound,
                        now 1 - [alpha_hat / (alpha_hat(time-1))]
            alpha [Vector]       : 1 - beta, Vector (per original implementation)
            alpha_hat [Vector]   : f(time) / f(0)
            f [Vector]           : cos((timeT/Time + s)/(1+s) * pi/2) ** 2
        
            s -> "In particular, we selected s such that √β0 was slightly smaller than the pixel
            bin size 1/127.5, which gives"
        '''
        steps = noise_step + 1
        x = torch.linspace(0, noise_step, steps, dtype = torch.float)
        cosine_function = torch.cos((((x/noise_step) + offset)/(1 + offset)) * (math.pi/2)) ** 2
        alpha_hat = cosine_function / cosine_function[0]
        beta = 1 - (alpha_hat[1:] / alpha_hat[:-1])
        beta = torch.clip(beta, 0, 0.999)

        return beta, 1.-beta, alpha_hat[:noise_step]

    #forward noising
    def noise_image(self, x, t):
        '''
            Xt = sqrt(alpha_hat)X0 + sqrt(1-alpha_hat)eplison, eplison ~ N(1, 0)
            t   : Random Time Step Vector [batchsize]
            x   : Input (image) Tensor [batchsize x channel x imgsize x imgsize]
        '''
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] #get alphahat for each image in batch
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None] #same thing
        e = torch.randn_like(x)
        e.to(self.device)
        #element wise addition and multiplication
        return sqrt_alpha_hat*x + sqrt_one_minus_alpha_hat*e, e

    #randomly sample some time step, return vector of size n
    def sample_timesteps(self, n):
        '''
            n   :size of the returned tensor
        '''
        return torch.randint(low=1, high=self.noise_step, size=(n,))
    
    #sampling aka generate image from noise
    #our input shape is 3 x 64 x 64, as 3 channel color (RBG) x height x width
    def sample(self, model, n, label, cfg_scale=3):
        '''
            n       : number of inputs (images)
            model   : diffusion model
        '''
        print(f"sampel {n} new images...")
        model.eval()

        #do not update the model
        with torch.no_grad():
            #creates a normal distribution of image size
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device) #[number of step x channel x imgsize x imgsize]
            for i in tqdm(reversed(range(1, self.noise_step)), position=0): #from 1000 to 1
                
                t = (torch.ones(n) * i).long().to(self.device) #[number of inputs] e.g. at time 10 with 5 img: [10, 10, 10, 10, 10]
                label = label.to(self.device)
                predicted_noise = model(x, t, label) #same shape as x
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, label, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]#same shape as x
                alpha_hat = self.alpha_hat[t][:, None, None, None]#same shape as x
                beta = self.beta[t][:, None, None, None]#same shape as x
                if i > 1:
                    noise = torch.rand_like(x) #Z
                else:
                    noise = torch.zeros_like(x)
                    
                #remove the predicted noise from guassian distribution
                x = 1/ torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()
        x = (x.clamp(-1, 1) + 1) /2 #clip out between -1, 1. Then make the output to between 0, 1
        x = (x * 255).type(torch.uint8) #this is to scale back to the pixel domain (RGB)
        return x