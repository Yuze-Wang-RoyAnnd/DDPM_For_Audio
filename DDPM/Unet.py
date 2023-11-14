import torch
import torch.nn as nn

#classifier free guidance implementation
class Unet_Conditional(nn.Module):
    '''
    the network takes a batch of noisy images of shape (batch_size, num_channels, height, width) 
    and a batch of noise levels of shape (batch_size, 1) as input, and returns a tensor of shape 
    (batch_size, num_channels, height, width) 
    '''
    def __init__(self, c_in=3, c_out=3, time_dim=256, numc_classes=None, device='cuda') -> None:
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        #encoder
        self.inc = DoubleConv(c_in, 64) #increase channel to 64
        self.down1 = Down(64, 128) #increase channel to 128, reduce image to 32 x 32
        self.sa1 = SelfAttention(128, 32) #channel remains the same
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)
        #bottleneck
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)
        #decoder
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if numc_classes is not None:
            self.label_embed = nn.Embedding(numc_classes, time_dim)
    
    #position encoding
    def pos_encoding(self, t, channels):

        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        '''
            x   : images
            t   : time steps, tensor with integer time value
            the trick is isntead of giving the time steps in the plain integer form, encode the time step
            using sin-cos embedding
        '''
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim) #output shape is [len(t) x self.time_dim]
        if y is not None:
            t += self.label_embed(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x =self.sa6(x)
        output =self.outc(x)
        return output



class DoubleConv(nn.Module):
    '''
    Two layer convolution block,
    the residual is just that the input tensor are add to the ouput of the double conv 
    increase our channel reduce spatial resolution, each filter channel might have learned different
    information
    '''
    def __init__(self, in_channels, out_channels, mid_channels = None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        '''
        GroupNorm:
        The input channels are separated into num_groups groups, 
        each containing num_channels / num_groups channels.
        num_channels must be divisible by num_groups.
        '''
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), #[batch, mid_channel, img, img]
            nn.GroupNorm(1, mid_channels), #normalize
            nn.GELU(), #activation
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), #[batch, out_channel, imgsize, imgsize]
            nn.GroupNorm(1, out_channels)
        )

    def forward(self, x):
        if self.residual:
            return nn.functional.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), #reduce the size by half
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels)
        )
        #resize our time embedding to fit the dimension, just linear projection
        self.embed_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )
    
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.embed_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) #project our time embedding
        return x + emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super().__init__()
        self.channels = channels
        self.size = size
        #channels = number of embeds,
        #num_heads = 4
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True) #input Query, Key, Value tensor
        self.ln = nn.LayerNorm([channels]) 
        self.ff_self = nn.Sequential(
            self.ln,
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        #flattens our x, then swap second and third dimension
        #e.g [8, 3, 64, 64] -> [8, 4096, 3]
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_val, _= self.mha(x_ln, x_ln, x_ln) #[batch, targetSequenceLength, embedding]
        attention_val = attention_val + x
        attention_val = self.ff_self(attention_val) + attention_val
        return attention_val.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)