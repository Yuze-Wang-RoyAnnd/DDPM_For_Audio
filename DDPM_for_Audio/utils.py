from matplotlib import pyplot as plt
from scipy import signal as sl
import torch
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation


from functools import partial

@staticmethod
def display_waveform(signal,sample_rate, text='Audio'):
    fig, ax = plt.subplots(1,1)
    ax.clear()
    fig.set_figwidth(20)
    fig.set_figheight(2)
    ax.scatter(np.arange(len(signal)),signal,s=1,marker='o',c='k')
    fig.suptitle(text, fontsize=16)
    ax.set_xlabel('time (secs)', fontsize=18)
    ax.set_ylabel('signal strength', fontsize=14)
    ax.axis([0,len(signal),-0.5,+0.5])
    time_axis,_ = plt.xticks()
    ax.set_xticks(time_axis[:-1],time_axis[:-1]/sample_rate)

@staticmethod
def display_waveform_hist(i, signal,sample_rate, fig, ax, text='Audio'):
    ax.clear()
    fig.set_figwidth(20)
    fig.set_figheight(2)
    ax.scatter(np.arange(len(signal[i])),signal[i],s=1,marker='o',c='k')
    fig.suptitle(f"{text}_at_{i}", fontsize=16)
    ax.set_xlabel('time (secs)', fontsize=18)
    ax.set_ylabel('signal strength', fontsize=14)
    ax.axis([0,len(signal[i]),-0.5,+0.5])
    time_axis,_ = plt.xticks()
    ax.set_xticks(time_axis[:-1],time_axis[:-1]/sample_rate)

@staticmethod
def save_as_ani(img_list, text, path):
    fig, ax = plt.subplots(1,1)
    ani = animation.FuncAnimation(fig, partial(display_waveform_hist, signal=img_list, sample_rate=22050, fig=fig, ax=ax, text=text), frames=len(img_list))
    ani.save(os.path.join(path, f'{text}.gif'), dpi=300, writer=animation.PillowWriter(fps=10))