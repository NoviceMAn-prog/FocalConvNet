from functools import partial

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, dim_out, 1),
        nn.BatchNorm2d(dim_out),
        nn.SiLU(),
        nn.Conv2d(dim_out, dim_out, 3, stride = stride, padding = 1, groups = dim_out),
        SqueezeExcitation(dim_out, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(dim_out, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

class FocalModulation(nn.Module):
    def __init__(self, dim, focal_window, focal_level, focal_factor=2, bias=True, proj_drop=0., use_postln=False):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln = use_postln

        self.f = nn.Linear(dim, 2*dim + (self.focal_level+1), bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()
                
        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, 
                    groups=dim, padding=kernel_size//2, bias=False),
                    nn.GELU(),
                    )
                )              
            self.kernel_sizes.append(kernel_size)          
        if self.use_postln:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        C = x.shape[-1]

        # pre linear projection
        x = self.f(x).permute(0, 3, 1, 2).contiguous()
        q, ctx, self.gates = torch.split(x, (C, C, self.focal_level+1), 1)
        
        # context aggreation
        ctx_all = 0 
        for l in range(self.focal_level):         
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx*self.gates[:, l:l+1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global*self.gates[:,self.focal_level:]

        # focal modulation
        self.modulator = self.h(ctx_all)
        x_out = q*self.modulator
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln:
            x_out = self.ln(x_out)
        
        # post linear porjection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0

        flops += N * self.dim * (self.dim * 2 + (self.focal_level+1))

        # focal convolution
        for k in range(self.focal_level):
            flops += N * (self.kernel_sizes[k]**2+1) * self.dim

        # global gating
        flops += N * 1 * self.dim 

        #  self.linear
        flops += N * self.dim * (self.dim + 1)

        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class FocalNetBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., 
                    act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                    focal_level=1, focal_window=3,
                    use_layerscale=False, layerscale_value=1e-4, 
                    use_postln=False,out_size=-1):
        super().__init__()
        self.dim = dim
        #self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio

        self.focal_window = focal_window
        self.focal_level = focal_level
        self.out_size = out_size
        self.norm1 = norm_layer([self.out_size,self.out_size,dim])
        self.modulation = FocalModulation(dim, proj_drop=drop, focal_window=focal_window, focal_level=self.focal_level, use_postln=use_postln)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer([self.out_size,self.out_size,dim])
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0    
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

        self.H = None
        self.W = None

    def forward(self, x):
       
        B, H,W, C = x.shape
        x = rearrange(x,'b c h w -> b h w c')
        shortcut = x
        
        
        x = self.norm1(x)
        
        
        x = self.modulation(x)
        
        x = shortcut + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        x = rearrange(x,'b h w c -> b c h w')
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, " \
               f"mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        
        flops += self.dim * H * W
        
        
        flops += self.modulation.flops(H*W)

        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class FocalConvNet(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        dim_head = 32,
        dim_conv_stem = None,
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        channels = 3,
        focal_level=[2, 2, 2, 2],
        focal_window=[3, 3, 3, 3],
        use_layerscale=False,
        layerscale_value=1e-4,
        use_postln=False
    ):
        super().__init__()
        assert isinstance(depth, tuple), 
        dim_conv_stem = default(dim_conv_stem, dim)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(channels, dim_conv_stem, 3, stride = 2, padding = 1),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding = 1)
        )
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.use_layerscale = use_layerscale
        self.layerscale_value = layerscale_value
        self.use_postln = use_postln

        

        num_stages = len(depth)
        self.num_layers = len(depth)
        

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])
        w = window_size
        out_size = [224,112,56,28,14,7]
        out_size_count = 1
        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim
                if is_first==1:
                    #print(out_size[out_size_count])
                    out_size_count+=1
                #print(layer_dim)
                block = nn.Sequential(
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample = is_first,
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate
                    ),
                    FocalNetBlock(layer_dim, mlp_ratio=4., drop=0., drop_path=0., 
                    act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                    focal_level=self.focal_level[ind], focal_window=self.focal_window[ind],
                    use_layerscale=self.use_layerscale, layerscale_value=self.layerscale_value, 
                    use_postln=self.use_postln,out_size = out_size[out_size_count])
                    
                )

                self.layers.append(block)
        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, x):
        x = self.conv_stem(x)
        
        for stage in self.layers:
            x = stage(x)
        return self.mlp_head(x)
model = FocalConvNet(
    num_classes = 11,
    dim_conv_stem = 64,               
    dim = 96,                         
    dim_head = 32,                    
    depth = (2, 2, 5, 2),             
    window_size = 7,                  
    mbconv_expansion_rate = 4,        
    mbconv_shrinkage_rate = 0.25,     
    dropout = 0.1, focal_level=[3, 3, 3, 3])                     
print("FocalConvNet created")
pytorch_total_params = sum(p.numel() for p in v.parameters())

print("Number of params",pytorch_total_params)
img = torch.randn(2, 3, 224, 224)

preds = v(img) # (2,11)
print(preds.shape)
