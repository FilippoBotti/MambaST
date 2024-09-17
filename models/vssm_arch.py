import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable
from timm.models.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
import numpy as np


NEG_INF = -1000000

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            is_cross=False,
            args=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.args=args
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_style_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        if is_cross:
            self.style_proj = (
                    nn.Linear(self.d_inner, (self.dt_rank + self.d_state), bias=False, **factory_kwargs),
                    nn.Linear(self.d_inner, (self.dt_rank + self.d_state), bias=False, **factory_kwargs),
                    nn.Linear(self.d_inner, (self.dt_rank + self.d_state), bias=False, **factory_kwargs),
                    nn.Linear(self.d_inner, (self.dt_rank + self.d_state), bias=False, **factory_kwargs),
                )
            self.x_proj = (
                    nn.Linear(self.d_inner, self.d_state, bias=False, **factory_kwargs),
                    nn.Linear(self.d_inner, self.d_state, bias=False, **factory_kwargs),
                    nn.Linear(self.d_inner, self.d_state, bias=False, **factory_kwargs),
                    nn.Linear(self.d_inner, self.d_state, bias=False, **factory_kwargs),
                )
            self.style_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.style_proj], dim=0))  # (K=4, N, inner)
            del self.style_proj
        else:
            self.x_proj = (
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            )
        
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core_zig_zag(self, x: torch.Tensor, style=None):
        B, C, H, W = x.shape
        L = H * W
        K = 4


        x_hw = x
        x_hw[:,:,:,1::2] = torch.flip(x_hw[:,:,:,1::2], dims=[2]) #flip column

        x_wh = x
        x_wh[:,:,1::2,:] = torch.flip(x_wh[:,:,1::2,:], dims=[3]) #flip row
        x_wh = torch.transpose(x_wh, dim0=2, dim1=3)

        x_hwwh = torch.stack([x_hw,x_wh], dim=1).view(B, 2, -1, L)

        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        
        if style is not None:
            style_hw = style
            style_hw[:,:,:,1::2] = torch.flip(style_hw[:,:,:,1::2], dims=[2]) #flip column

            style_wh = style
            style_wh[:,:,1::2,:] = torch.flip(style_wh[:,:,1::2,:], dims=[3]) #flip row
            style_wh = torch.transpose(style_wh, dim0=2, dim1=3)

            style_hwwh = torch.stack([style_hw,style_wh], dim=1).view(B, 2, -1, L)

            style = torch.cat([style_hwwh, torch.flip(style_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

            s_dbl = torch.einsum("b k d l, k c d -> b k c l", style.view(B, K, -1, L), self.style_proj_weight)
            
            dts, Bs = torch.split(s_dbl, [self.dt_rank, self.d_state], dim=2)
            Cs = x_dbl
                
        else:
            dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        
        if style is not None:
            style = style.float().view(B, -1, L)
            
            out_y = self.selective_scan(
            style, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        else:
            out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
            
        out_hw = out_y[:, 0].view(B, -1, H, W)
        out_hw[:,:,:,1::2] = torch.flip(out_hw[:,:,:,1::2], dims=[2]) #flip column

        out_wh = out_y[:, 1].view(B, -1, W, H)
        out_wh = torch.transpose(out_wh, dim0=2, dim1=3)
        out_wh[:,:,1::2,:] = torch.flip(out_wh[:,:,1::2,:], dims=[3]) #flip row

        out_inv = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        out_inv_hw = out_inv[:, 0].view(B, -1, H, W)
        out_inv_hw[:,:,:,1::2] = torch.flip(out_inv_hw[:,:,:,1::2], dims=[2]) #flip column

        out_inv_wh = out_inv[:, 1].view(B, -1, W, H)
        out_inv_wh = torch.transpose(out_inv_wh, dim0=2, dim1=3)
        out_inv_wh[:,:,1::2,:] = torch.flip(out_inv_wh[:,:,1::2,:], dims=[3]) #flip row

        return out_hw, out_wh, out_inv_hw, out_inv_wh
    
    def forward_core(self, x: torch.Tensor, style=None):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        
        if style is not None:
            style = torch.stack([style.view(B, -1, L), torch.transpose(style, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
            style = torch.cat([style, torch.flip(style, dims=[-1])], dim=1) # (1, 4, 192, 3136)
            s_dbl = torch.einsum("b k d l, k c d -> b k c l", style.view(B, K, -1, L), self.style_proj_weight)

            dts, Bs = torch.split(s_dbl, [self.dt_rank, self.d_state], dim=2)
            Cs = x_dbl
                
        else:
            dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        
        if style is not None:
            style = style.float().view(B, -1, L)
            #change hereeeeeeee xs and style
            out_y = self.selective_scan(
            style, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        else:
            out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
            
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, style=None, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        
        x, z = xz.chunk(2, dim=-1)
        if style is not None:
            style = self.in_style_proj(style)
            
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        
        if style is not None:
            style = style.permute(0, 3, 1, 2).contiguous()
            style = self.act(self.conv2d(style))
            y1, y2, y3, y4 = self.forward_core(x, style=style)
        else:
            y1, y2, y3, y4 = self.forward_core(x)
            
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            is_cross=False,
            args=None,
            **kwargs,
    ):
        super().__init__()
        self.args = args
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, is_cross=is_cross, args=args, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)


    def forward(self, input, style=None):
        # x [B,HW,C]
        input = rearrange(input, 'l b d -> b l d')
        if style is not None:
            style = rearrange(style, 'l b d -> b l d')
            rnd = torch.rand(style.shape[1])
            indexes = torch.argsort(rnd)
            
        if self.args is not None and self.args.rnd_style:
            style = style[:,indexes,:]

        B, L, C = input.shape
        input = input.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).contiguous()  # [B,H,W,C]
        if style is not None:
            style = style.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).contiguous()  # [B,H,W,C]
            
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x, style=style))
        x = x.view(B, -1, C).contiguous()
        # norm here used if there's no channel attention
        x = self.norm(x)
        x = rearrange(x, 'b l d -> l b d')
        return x

