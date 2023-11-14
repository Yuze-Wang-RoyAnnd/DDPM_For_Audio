import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch import optim
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
import numpy as np
import copy
from ForwardDiffusion import Diffusion
import Unet

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1)
    ], dim=-1).permute(1, 2, 0).cpu())
    plt.show()

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    '''
        length of dataloader is the size of the folder with classs
        Image Tensor, number of images = next(iter(dataloader))
    '''

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),
        torchvision.transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

#setup folders
def setup_loggin(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

class myarg():
    def __init__(self):
        self.run_name = "DDPM_Unconditional"
        self.epochs = 100
        self.batch_size  = 16
        self.img_size = 64
        self.dataset_path = "/root/autodl-tmp/DDPM/cifar10-64/train"
        self.device = 'cuda'
        self.lr = 3e-4
        self.num_classes = 10

'''class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())'''

from torch.utils.tensorboard import SummaryWriter
def train(args):
    setup_loggin(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = Unet.Unet_Conditional(numc_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, device=device)
    l = len(dataloader)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    #ema = EMA(beta=0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch}")
        pbar = tqdm(dataloader) #each iteration return a [batchsize x channel x imgsize x imgsize] tensor
        for i, (images, label) in enumerate(pbar):
            '''
            forward step: get bunch of random time steps, noise the images to that time steps
            '''
            images = images.to(device) #[batchsize x channel (RGB) x imgsize x imgsize]
            label = label.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device) #[batchsize]
            x_t, noise = diffusion.noise_image(images, t) #add noise to image
            if np.random.random() < 0.1: #cfg training, at random, drop the label and predict the noise
                label = None
            '''
            reverse step: get the predicted noise, takes gradient of
            the mean square root loss between noise and predicted noise
            '''
            #each image within x_t corrospond to a item in t
            predicted_noise = model(x_t, t, label)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
        if epoch % 10 == 0:
            sampled_images = diffusion.sample(model, n=images.shape[0], label=label)
            save_images(sampled_images, os.path.join("/root/autodl-tmp/DDPM/results/", args.run_name, f"{epoch}.jpg"))
            torch.save(model.state_dict(), os.path.join("/root/autodl-tmp/DDPM/models/", args.run_name, f"ckpt.pt"))


if __name__ == "__main__":
    import torch
    torch.cuda.empty_cache()
    arg = myarg()
    train(arg)