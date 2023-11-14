import torch
import torchaudio
import torch.utils.data
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

batchSize = 4
def collate_fn(data):
    '''
       data: some audios
       we randomly choose audio frames 1, 2, 3, 4, 5 seconds
    '''
    batch_range = len(data[0])//44100//batchSize
    #44.1KHz :
    choosenRange = np.random.choice([1, 2, 3, 4, 5])
    d_stack = []
    for i in range(batchSize):
        randomStart = int(np.random.uniform(batch_range*i, batch_range*(i+1)-choosenRange))
        d_stack.append(data[:, randomStart*44100:randomStart*44100+choosenRange*44100])
    output = torch.stack(d_stack)
    return output


class PianoSet(Dataset):
    def __init__(self, path_to_data, cycle=2) -> None:
        '''
        cycle control how many time we get data from the same data set
        since the random choose does not conver everything in one audio file
        '''
        self.path = path_to_data
        self.data = next(os.walk(path_to_data), (None, None, []))[2]
        self.cycle = cycle
    
    def __len__(self):
        return len(self.data) * self.cycle

    def __getitem__(self, idx):
        if idx >= len(self.data):
            #cycle back
            idx = idx-len(self.data)
        audio, _ = torchaudio.load(os.path.join(self.path, self.data[idx]))
        #downmix to single channel
        audio = torch.mean(audio, dim=0, keepdim=True)
        return audio
