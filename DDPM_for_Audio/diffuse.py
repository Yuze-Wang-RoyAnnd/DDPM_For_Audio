import torch
from tqdm import tqdm
class forwardDiff:
    def __init__(self, beta_min, beta_max, diffusion_step=500, device='cpu') -> None:
        self.diffusion_step = diffusion_step
        self.beta, self.alpha, self.alpha_hat = self.noise_schedule(beta_min, beta_max)
        self.device = device
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_hat = self.alpha_hat.to(device)


    def noise_schedule(self, beta_min, beta_max):
        beta = torch.linspace(beta_min, beta_max, self.diffusion_step, dtype=torch.float)
        alpha = 1 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)
        return beta, alpha, alpha_hat

    #audio = [batch x channel x feature]
    def noise_audio(self, audio, time):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[time])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[time])[:, None, None]
        e = torch.rand_like(audio)
        return sqrt_alpha_hat * audio + sqrt_one_minus_alpha_hat * e, e
    
    def sample_timesteps(self, n):
        return torch.randint(low=1 ,high=self.diffusion_step, size=n)
    
    def sample(self, model):
        model.eval()

        with torch.no_grad():
            x = torch.randn((3, 1, 441000)).to(self.device)
            for i in tqdm(reversed(range(1, self.diffusion_step)), position=0):
                pass
