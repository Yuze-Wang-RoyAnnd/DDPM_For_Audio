from matplotlib import pyplot as plt
from scipy import signal as sl
import torch
import numpy as np

@staticmethod
def visualize_noising(audio, timeSteps:list, dif):
    fig, ax = plt.subplots(len(timeSteps))
    noises, _ = dif.noise_audio(audio, timeSteps)
    for i, n in enumerate(noises):
        ax[i].plot(torch.arange(len(n[0])), n[0, :])
    plt.show()

@staticmethod
def visualize_noising_as_mel(audio, timeSteps:list, dif):
    fig, ax = plt.subplots(len(timeSteps))
    noises, _ = dif.noise_audio(audio, timeSteps)
    for i, n in enumerate(noises):
        f, t, Sxx = sl.spectrogram(n, 44100, window='hamming',nperseg=512, noverlap=256, scaling='density')

        ax[i].pcolormesh(t, f[:100], 20*np.log10(Sxx[0, :100, :]), shading='gouraud')
    plt.show()