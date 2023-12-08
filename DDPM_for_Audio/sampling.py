import numpy as np
import os
import torch
import librosa
import soundfile
from tqdm import tqdm

from argparse import ArgumentParser

from diffwave.src.diffwave.params import AttrDict, params as base_params
from diffwave.src.diffwave.model import DiffWave

import torch.nn as nn
from utils import display_waveform, save_as_ani
import math


models = {}


def noise_audio(audio, alpha_tensor, time, device):
    alpha = alpha_tensor[[time]].unsqueeze(1) #1 x 1
    sqrt_alpha = alpha**0.5 # 1 x 1
    noisy_audio = sqrt_alpha * audio
    if time > 0:
        noise = torch.from_numpy(np.random.randn(*audio.shape).astype(np.float32)).to(device)
        noisy_audio += (1.0 - alpha)**0.5 * noise    
    return noisy_audio

def denoise_audio(audio, predicted,alpha, alpha_cum, beta, step, device):
    c1 = 1 / alpha[step]**0.5
    c2 = beta[step] / (1 - alpha_cum[step])**0.5
    audio = c1 * (audio - c2 * predicted.squeeze(1))
    if step > 0:
        noise = torch.from_numpy(np.random.randn(*audio.shape).astype(np.float32)).to(device) #torch.randlike is not consisten!!!, this is a workaround
        sigma = ((1.0 - alpha_cum[step-1]) / (1.0 - alpha_cum[step]) * beta[step])**0.5
        audio += sigma * noise
    return audio

def smple(model, device):
    with torch.no_grad():
        beta = torch.tensor(np.array(model.params.noise_schedule).astype(np.float32)).to(device)
        alpha = 1 - beta
        alpha_cum = torch.cumprod(alpha, dim=0)
        time_schedule = np.arange(0, len(beta), dtype=np.float32)
        target = torch.randn(1, base_params.audio_len, device=device)
        tq_d = tqdm(range(len(alpha)-1, -1, -1), position=0)
        for n in tqdm(tq_d):
            p_noise = model(target, torch.tensor([time_schedule[n]], device=device), None)
            target = denoise_audio(target, p_noise, alpha, alpha_cum, beta, n, device)
            target = torch.clamp(target, -1.0, 1.0) #renormalize
        for i in range(5):
            p_noise = model(target, torch.tensor([time_schedule[n]], device=device), None)
            target = denoise_audio(target, p_noise, alpha, alpha_cum, beta, n, device)
            target = torch.clamp(target, -1.0, 1.0) #renormalize

    return target


def resampling_extension(model, reference: torch.tensor, device, track_history=False):
    if track_history:
        target_through_time = []
    with torch.no_grad():
        beta = torch.tensor(np.array(model.params.noise_schedule).astype(np.float32)).to(device)
        alpha = 1.0 - beta
        alpha_cum = torch.cumprod(alpha, dim=0)
        time_schedule = np.arange(0, len(beta), dtype=np.float32)
        target = torch.randn(1, base_params.audio_len, device=device)
        rf = reference.clone()
        n = len(alpha)
        jump_length = 10
        jump_range = torch.arange(((len(alpha) - jump_length) // jump_length)) * jump_length
        #results in [0, 10, 20, 30, ...]
        jump_range = jump_range.tolist()
        print(jump_range)
        U = 10
        pbar = tqdm(total=200)
        while n >= 1:
            pbar.set_postfix_str(f'n: {n}, diffusion step: {n-1}')
            rf1 = noise_audio(rf, alpha_cum, n-1, device) #noise to match the time step (x^known)
            #denoise (x^unknown)
            p_noise = model(target, torch.tensor([time_schedule[n-1]], device=device), None)
            target = denoise_audio(target, p_noise, alpha, alpha_cum, beta, n-1, device)
            target = torch.cat((rf1, target[:, -(base_params.audio_len - rf1.shape[1]):]), dim=1)
            target = torch.clamp(target, -1.0, 1.0) #renormalize
            if n in jump_range: #ignore the first #jump_length# steps
                curr_n = n
                for _ in range(U):
                    #re-noise for jump_length step
                    for _ in range(jump_length):
                        if n > 1:
                            target = noise_audio(target, alpha, curr_n-2, device)
                        curr_n += 1
                    for _ in range(jump_length):
                        p_noise = model(target, torch.tensor([time_schedule[curr_n-1]], device=device), None)
                        target = denoise_audio(target, p_noise, alpha, alpha_cum, beta, curr_n-1, device)
                        target = torch.clamp(target, -1.0, 1.0) #renormalize
                        curr_n -= 1
            n -= 1
            pbar.update(1)
            if track_history:
                target_through_time.append(target.clone().squeeze(0).detach().cpu())
        for _ in range(3):
            target = noise_audio(target, alpha_cum, 0, device)
            p_noise = model(target, torch.tensor([time_schedule[n]], device=device), None)
            target = denoise_audio(target, p_noise, alpha, alpha_cum, beta, n, device)
            target = torch.clamp(target, -1.0, 1.0) #renormalize
    if track_history:
        return target, target_through_time
    else:
        return target


def interpolation_extention(model, reference: torch.tensor, device, track_history=False):
    '''
        model       : diffwave model
        reference   : base sound file to extends from
        device      : cuda/cpu/mps
        tack_history: whether to generate diffusion history (for visualization)
    '''
    if track_history:
        target_through_time = []
    with torch.no_grad():
        beta = torch.tensor(np.array(model.params.noise_schedule).astype(np.float32)).to(device)
        alpha = 1 - beta
        alpha_cum = torch.cumprod(alpha, dim=0)
        time_schedule = np.arange(0, len(beta), dtype=np.float32)
        target = torch.randn(1, base_params.audio_len, device=device)
        lmbda = torch.arange(len(beta), dtype=torch.float32)
        lmbda.apply_(lambda x: (((0.3704*math.e)**(0.5*x))-1)**4) #set lmbda schedule
        lmbda[-1] = 1
        rf = reference.clone()
        tq_d = tqdm(range(len(alpha) - 1, -1, -1), position=0)
        for i, n in enumerate(tq_d):
            rf1 = noise_audio(rf, alpha_cum, n, device)
            target = torch.cat(((((1-lmbda[i]) * rf1) + (lmbda[i] * target[:, :reference.shape[1]])), \
                    target[:, -(base_params.audio_len - reference.shape[1]):]), dim=1)
            p_noise = model(target, torch.tensor([time_schedule[n]], device=device), None)
            target = denoise_audio(target, p_noise, alpha, alpha_cum, beta, n, device)
            target = torch.clamp(target, -1.0, 1.0) #renormalize
            if track_history:
                target_through_time.append(target.clone().squeeze(0).detach().cpu())
            tq_d.set_postfix_str(s=f"lambda: {lmbda[i]}", refresh=True)
        #denoise one more time
        for _ in range(5):
            p_noise = model(target, torch.tensor([time_schedule[n]], device=device), None)
            target = denoise_audio(target, p_noise, alpha, alpha_cum, beta, n, device)
            target = torch.clamp(target, -1.0, 1.0) #renormalize
    if track_history:
        return target, target_through_time
    else:
        return target


def normal_ext(model, reference: torch.tensor, device, track_history=False):
    target_through_time = []
    with torch.no_grad():
        beta = torch.tensor(np.array(model.params.noise_schedule).astype(np.float32)).to(device)
        alpha = 1 - beta
        alpha_cum = torch.cumprod(alpha, dim=0)
        time_schedule = np.arange(0, len(beta), dtype=np.float32)
        target = torch.randn(1, base_params.audio_len, device=device)
        rf = reference.clone()
        tq_d = tqdm(range(len(alpha) - 1, -1, -1), position=0)
        for i, n in enumerate(tq_d):
            rf1 = noise_audio(rf, alpha_cum, n, device)
            target = torch.cat((rf1, \
                    target[:, -(base_params.audio_len - reference.shape[1]):]), dim=1)
            p_noise = model(target, torch.tensor([time_schedule[n]], device=device), None)
            target = denoise_audio(target, p_noise, alpha, alpha_cum, beta, n, device)
            target = torch.clamp(target, -1.0, 1.0) #renormalize
            if track_history:
                target_through_time.append(target.clone().squeeze(0).detach().cpu())
        for _ in range(5):
            p_noise = model(target, torch.tensor([time_schedule[n]], device=device), None)
            target = denoise_audio(target, p_noise, alpha, alpha_cum, beta, n, device)
            target = torch.clamp(target, -1.0, 1.0) #renormalize
    if track_history:
        return target, target_through_time
    else:
        return target


def setup(args):
    if isinstance(args, ArgumentParser): #as CLI
        model_dir = args.parse_args().model_dir
    elif isinstance(args, dict): #as API
        model_dir = args['model_dir']
    else:
        raise TypeError("Unrecongized argument")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if not model_dir in models:
        if os.path.exists(f'{model_dir}/weights.pt'):
            checkpoint = torch.load(f'{model_dir}/weights.pt', map_location=torch.device(device))
        else:
            checkpoint = torch.load(model_dir, map_location=torch.device(device))
        model = DiffWave(AttrDict(base_params)).to(device)
        if hasattr(model, 'module') and isinstance(model.module, nn.Module):
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        model.eval()
        models[model_dir] = model
    model = models[model_dir]
    model.params.override(base_params)
    return model, device

def infer(model, extent, ground_file=None, device='mps', extension_seconds = 3):
    if ground_file:
        reference, fs = librosa.load(ground_file)
        reference = torch.tensor(reference.astype(np.float32)).unsqueeze(0).to('mps')
        assert fs == model.params.sample_rate, f"sample rate mismatch, expected {model.params.sample_rate} but got {fs}"
    else:
        reference = smple(model, device)
    output = reference.clone()


    for i in range(extension_seconds):
        print(f'Extending {i+1} second(s)')
        sample = extent(model, output[:, -88200:], device)
        output = torch.cat((output[:, :-88200], sample), dim=1)
    
    return output
    #generate a 10 second audio, each extension pass generate an audio that is 1 second longer



def main(parser):
    model, device = setup(parser)
    args = parser.parse_args()
    match args.extent:
        case 'interpolation_extention':
            algo = interpolation_extention
        case 'resampling_extension':
            algo = resampling_extension
        case 'normal_extention':
            algo = normal_ext
        case _:
            raise TypeError('Unrecongized extention algorithm')

    wav, fs = infer(model, algo, args.ground_file, device, int(args.seconds))
    soundfile.write(args.output, wav, base_params.sample_rate)


if __name__ == '__main__':
  parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('model_dir',
      help='directory containing a trained model (or full path to weights.pt file)')
  parser.add_argument('--ground_file',
      help='path to audio file that we want to generated from, if not provided, generate from static')
  parser.add_argument('--output', default='output.wav',
      help='output file name')
  parser.add_argument('--extent',
      help='inpaint algorithm you want to use, default=interpolation_extention', 
      default='interpolation_extention')
  parser.add_argument('--seconds',
      help='how long do you want to extends', default=5)
  main(parser)