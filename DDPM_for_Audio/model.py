import torch
import torch.nn as nn

class unet(nn.Module):
    def __init__(self, inchannels, batch_size =8, time_dim=256, device='cuda') -> None:
        super().__init__()
        self.register_buffer("time_dim", time_dim)
        self.register_buffer("batch", batch_size)
        self.register_buffer("device", device)
        self.encode = nn.ModuleList([
            nn.Conv1d(inchannels, 32, kernel_size=7), #[batch x channel x inputfeature] -> [batch x 32 x inputfeature]
            convoBlock(32, 64, 2, direction='down'),
            corssAttention(64),
            convoBlock(64, 128, 4, direction='down'),
            corssAttention(128),
            convoBlock(128, 256, 5, direction='down'),
            corssAttention(256),
            convoBlock(256, 512, 8, direction='down'),
        ])

        #[batch x 256 x inputfeature/(2*4)]
        #lstm
        self.lstm = nn.Sequential(
            nn.LSTM(512, 512, 2, batch_first=True),
            nn.Linear(2 * 512, 512)
        )


        self.bottleneck = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=7),
            nn.Conv1d(512, 512, kernel_size=7),
            nn.Conv1d(512, 512, kernel_size=7)
        )

        self.decode = nn.ModuleList([
            convoBlock(512, 256, 8, direction='up'),
            corssAttention(256),
            convoBlock(256, 128, 5, direction='up'),
            corssAttention(128),
            convoBlock(128, 64, 4, direction='up'),
            corssAttention(64),
            convoBlock(64, 32, 2, direction='up'),
            nn.Conv1d(32, inchannels, kernel_size=7)
        ])

    
    #position encoding
    def pos_encoding(self, input, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(input.repeat(1, channels // 2) * inv_freq) 
        pos_enc_b = torch.cos(input.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc #[channels x len(t)]


    def forward(self, x, t, source):
        #t is the diffusion step
        #souce the relative position of the frame in the whole sound
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        source = self.pos_encoding(source, self.time_dim)
        source += t

        h_0, c_0 = torch.zeros(2, self.batch, 512), torch.zeros(2, self.batch, 512)
        #encode
        for module in self.encode:
            if isinstance(module, corssAttention):
                x = module(x, source)
            else:
                x = module(x)

        #lstm
        h_0_1, c_0_1 = h_0.detach().clone().to(self.device), c_0.detach().clone().to(self.device)
        x, _, _ = self.lstm(x, h_0_1, c_0_1)

        #bottle neck
        x = self.bottleneck(x)

        #lstm
        h_0_2, c_0_2 = h_0.detach().clone().to(self.device), c_0.detach().clone().to(self.device)
        x, _, _ = self.lstm(x, h_0_2, c_0_2)

        #decode
        for module in self.decode:
            if isinstance(module, corssAttention):
                x = module(x, source)
            else:
                x = module(x)

        return x


class corssAttention(nn.Module):
    def __init__(self, channels, embed_dim=256) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True) #each head produce dim = channel // 4
        self.ln = nn.LayerNorm([channels]) 
        self.ff = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

        #resize information embeding
        self.project = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, channels)
        )

    def forward(self, x, source):
        x = x.swapaxes(1, 2) #[batch x input x channel]
        x_ln = self.ln(x)

        source = self.project(source)        
        attention_val, _=self.mha(x_ln, source, source)
        attention_val = attention_val + x
        attention_val = self.ff(attention_val) + attention_val
        return attention_val.swapaxes(2, 1)



class convoBlock(nn.Module):
    def __init__(self, inchannels, outchannel, down_stride, direction) -> None:
        super().__init__()
        #residual unit
        self.double_conv = nn.Sequential(
            nn.GroupNorm(inchannels), #normalize
            nn.ELU(), #activation
            nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1, bias=False), 
            nn.GroupNorm(inchannels),
            nn.ELU(),
            nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1, bias=False)
        )
        if direction == 'down':
            #downsampling, increase channels by 2
            self.resample = nn.Sequential(
                nn.Conv1d(inchannels, outchannel, kernel_size=down_stride*2, stride=down_stride),
                nn.LayerNorm(inchannels), #normalize
                nn.GELU() #activation
            )
        else:
            #upsampling, decrease channels by 2
            self.resample = nn.Sequential(
                nn.ConvTranspose1d(inchannels, outchannel, kernel_size=down_stride*2, stride=down_stride),
                nn.LayerNorm(inchannels), #normalize
                nn.GELU() #activation
            )
    def forward(self, x):
        output = self.double_conv(x)
        output = output + x
        return self.resample(output)

