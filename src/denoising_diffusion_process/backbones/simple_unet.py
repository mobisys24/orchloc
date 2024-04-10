import torch
from torch import nn
import math
from einops import rearrange
import math
from inspect import isfunction


def exists(x):

    #  If x is not None, it returns True; otherwise, it returns False. 
    return x is not None


def default(val, d):

    if exists(val):
        return val
    
    # isfunction returns True when passed a function, and False when passed a non-function
    return d() if isfunction(d) else d


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()

        self.fn = fn


    def forward(self, x, *args, **kwargs):

        return self.fn(x, *args, **kwargs) + x


class m_Linear(nn.Module):

    def __init__(self, size_in, size_out):
        super().__init__()

        self.size_in, self.size_out = size_in, size_out

        # Creation
        self.weights_real = nn.Parameter(torch.randn(size_in, size_out, dtype=torch.float32))
        self.weights_imag = nn.Parameter(torch.randn(size_in, size_out, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(2, size_out, dtype=torch.float32))

        # Initialization
        nn.init.xavier_uniform_(self.weights_real, gain=1)
        nn.init.xavier_uniform_(self.weights_imag, gain=1)
        nn.init.zeros_(self.bias)


    def swap_real_imag(self, x):
        h = x

        h = h.flip(dims=[-2])

        h = h.transpose(-2, -1)

        # performs an element-wise multiplication along the last dimension
        h = h * torch.tensor([-1, 1]).cuda()

        h = h.transpose(-2, -1)

        return h


    def forward(self, x):
        h = x

        h1 = torch.matmul(h, self.weights_real)

        h2 = torch.matmul(h, self.weights_imag)

        h2 = self.swap_real_imag(h2)

        h = h1 + h2

        h = torch.add(h, self.bias)

        return h


class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)


    def forward(self, x):
        b, c, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) w -> b h c w', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c w -> b (h c) w', h=self.heads, w=w)

        return self.to_out(out)


class SinusoidalPositionEmbeddings(nn.Module):
    """
        Based on transformer-like embedding from 'Attention is all you need'
        Note: 10,000 corresponds to the maximum sequence length
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim


    def forward(self, time):
        assert len(time.shape) == 1

        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings


class UpBlock(nn.Module):

    def __init__(self, input_length):
        super().__init__()

        self.leng = input_length
        self.up = m_Linear(input_length, input_length * 2)


    def forward(self, x):
        _, ch, _ = x.shape

        complex_di = 2

        x = x.reshape((-1, complex_di, ch, self.leng))   # (@ * 2, ch, dim) => (@, 2, ch, dim)
        x = x.permute(0, 2, 1, 3)                        # (@, 2, ch, dim) => (@, ch, 2, dim)

        x = self.up(x)                                   # (@, ch, 2, dim) => (@, ch, 2, dim * 2)
        x = x.permute(0, 2, 1, 3)                        # (@, ch, 2, dim * 2) => (@, 2, ch, dim * 2)
        x = x.reshape((-1, ch, self.leng * 2))           # (@, 2, ch, dim * 2) => (@ * 2, ch, dim * 2)

        return x
    

class DownBlock(nn.Module):

    def __init__(self, input_length):
        super().__init__()

        self.leng = input_length

        self.down = m_Linear(input_length, input_length // 2)

    def forward(self, x):

        _, ch, _ = x.shape
        complex_di = 2
        x = x.reshape((-1, complex_di, ch, self.leng))   # (@ * 2, ch, dim) => (@, 2, ch, dim)
        x = x.permute(0, 2, 1, 3)                        # (@, 2, ch, dim) => (@, ch, 2, dim)

        x = self.down(x)                                 # (@, ch, 2, dim) => (@, ch, 2, dim // 2)
        x = x.permute(0, 2, 1, 3)                        # (@, ch, 2, dim // 2) => (@, 2, ch, dim // 2)
        x = x.reshape((-1, ch, self.leng // 2))          # (@, 2, ch, dim // 2) => (@ * 2, ch, dim // 2)

        return x


class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        # dim are channels 
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))


    def forward(self, x):
        # computes the mean and variance of the input x along the channel dimension (dim=1)
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)

        # normalizes x
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)


    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# building block modules for signal (containg location info) and timestep
class ComplexTimeBlock(nn.Module):

    def __init__(self, 
                 dim, 
                 dim_out, 
                 *, 
                 time_emb_dim=None, 
                 mult=2, 
                 norm=True):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim)
        ) if exists(time_emb_dim) else None
  
        self.ds_conv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv1d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(dim_out * mult, dim_out, 3, padding=1)
            )

        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()


    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, 'b c -> b c 1')

        h = self.net(h)

        residual_x = self.res_conv(x)

        return h + residual_x


##### Main Model #####
class UnetComplexBlock(nn.Module):

    def __init__(self, 
                 dim, 
                 loc_dim=7,
                 channels=2,
                 dim_mults=(1, 2, 4, 8)):
        super().__init__()

        self.leng = dim
        time_dim = dim

        # if dim = 16, [2, 16, 32, 64, 128]
        dims = [channels, *map(lambda m: dim * m, dim_mults)]

        # [(2, 16), (16, 32), (32, 64), (64, 128)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # dim * (2 ^ 0), dim * (2 ^ 1), dim * (2 ^ 2), dim * (2 ^ 3): [16, 32, 64, 128]
        dim_list_sample = [dim * int(math.pow(2, scale))  for scale in range(len(dim_mults))]

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.class_emb = nn.Sequential(
            nn.Linear(loc_dim, dim), 
            nn.GELU(),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # [(2, 16), (16, 32), (32, 64), (64, 128)]
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            length_temp = dim_list_sample[ind] if not is_last else None  # [16, 32, 64, 128]

            self.downs.append(nn.ModuleList([
                ComplexTimeBlock(dim_in, dim_out, time_emb_dim=time_dim, norm=(ind!=0)),
                ComplexTimeBlock(dim_out, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                UpBlock(length_temp) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ComplexTimeBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ComplexTimeBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        # [(64, 128), (32, 64), (16, 32)]
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):  # ind: 0, 1, 2
            # is_last will not go to True, always be False
            is_last = ind >= (num_resolutions - 1)
            length_temp = dim_list_sample[len(dim_list_sample) - ind - 1] if not is_last else None

            self.ups.append(nn.ModuleList([
                ComplexTimeBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                ComplexTimeBlock(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                DownBlock(length_temp) if not is_last else nn.Identity()
            ]))

        out_dim = channels - 1  # out_dim is channels

        self.final_conv = nn.Sequential(
            ComplexTimeBlock(dim, dim),
            nn.Conv1d(dim, out_dim, 1),
        )


    def forward(self, feature_x, time, location):
        # feature_x: torch.Size([@, 2, 16])
        # time: torch.Size([@])
        # location: torch.Size([@, 7])

        time_2 = torch.cat((time, time), dim=0)                  # (@, ) => (@ * 2, )
        t = self.time_mlp(time_2)                                # (@ * 2, ) => (@ * 2, dim)

        class_cond = self.class_emb(location)                    # (@, 7) => (@, dim)
        class_cond = class_cond.unsqueeze(dim=1)                 # (@, dim) => (@, 1, dim)
        class_cond = torch.cat((class_cond, class_cond), dim=1)  # (@, 1, dim) => (@, 2, dim)
        class_cond = class_cond.unsqueeze(dim=1)                 # (@, 2, dim) => (@, 1, 2, dim)

        feature_x = feature_x.unsqueeze(dim=1)                   # (@, 2, dim) => (@, 1, 2, dim)

        x = torch.cat((feature_x, class_cond), dim=1)
        [channle, complex_dim, length] = x.shape[-3: ]

        x = x.permute(0, 2, 1, 3)
        x = x.reshape((-1, channle, length))                     # (@, 2, 2, dim) => (@ * 2, 2, dim)

        h = []
        for convnext, convnext2, attn, upsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = upsample(x)

        x = self.mid_block1(x, t)   # (@ * 2, 128, dim * 8)
        x = self.mid_attn(x)        # (@ * 2, 128, dim * 8)
        x = self.mid_block2(x, t)   # (@ * 2, 128, dim * 8)

        for convnext, convnext2, attn, downsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)  # R0: (@ * 2, 128 * 2, dim * 8)
            x = convnext(x, t)               
            x = convnext2(x, t)
            x = attn(x)
            x = downsample(x)        

        # (@ * 2, 16, dim) => (@ * 2, 1, dim): (@ * 2, channel, dim)
        out = self.final_conv(x)

        out = out.reshape((-1, complex_dim, 1, self.leng))    # (@ * 2, 1, dim) => (@, 2, 1, dim)

        out = out.permute(0, 2, 1, 3)                         # (@, 2, 1, dim) => (@, 1, 2, dim)
        out = out.squeeze(dim=1)                              # (@, 1, 2, dim) => (@, 2, dim)

        return out




