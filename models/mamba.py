import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
device = torch.device("cuda")
from .vssm_arch import VSSBlock
# from .single_direction_vssm_arch import VSSBlock
# from .double_direction_vssm_arch import VSSBlock


class Mamba(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=512, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, args=None):
        super().__init__()
        
        self.args = args
        
        if self.args is not None:
            encoder_layer = VSSBlock(
                hidden_dim=d_model,
                drop_path=0,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=self.args.d_state,
                input_resolution=self.args.img_size)
            
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder_c = Encoder(encoder_layer, num_encoder_layers, encoder_norm, args=self.args)
        self.encoder_s = Encoder(encoder_layer, num_encoder_layers, encoder_norm, args=self.args)

        if self.args is not None:
            decoder_layer = VSSBlock(
                hidden_dim=d_model,
                drop_path=dropout,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=self.args.d_state,
                input_resolution=self.args.img_size,
                is_cross=True,
                args=args)
        
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec, args=args)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.new_ps = nn.Conv2d(512 , 512 , (1,1))
        self.averagepooling = nn.AdaptiveAvgPool2d(18)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, style, mask , content, pos_embed_c, pos_embed_s):
        # content-aware positional embedding
        content_pool = self.averagepooling(content)       
        pos_c = self.new_ps(content_pool)
        pos_embed_c = F.interpolate(pos_c, mode='bilinear',size= style.shape[-2:])

        ###flatten NxCxHxW to HWxNxC     
        style = style.flatten(2).permute(2, 0, 1)
        if pos_embed_s is not None:
            pos_embed_s = pos_embed_s.flatten(2).permute(2, 0, 1)
      
        content = content.flatten(2).permute(2, 0, 1)
        if pos_embed_c is not None:
            pos_embed_c = pos_embed_c.flatten(2).permute(2, 0, 1)
     
        style = self.encoder_s(style, src_key_padding_mask=mask, pos=pos_embed_s)
        content = self.encoder_c(content, src_key_padding_mask=mask, pos=pos_embed_c)
        hs = self.decoder(content, style, memory_key_padding_mask=mask,
                          pos=pos_embed_s, query_pos=pos_embed_c)[0]
        
        ### HWxNxC to NxCxHxW to
        N, B, C= hs.shape          
        H = int(np.sqrt(N))
        hs = hs.permute(1, 2, 0)
        hs = hs.view(B, C, -1,H)
        return hs


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, args=None):
        super().__init__()
        self.args = args
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        
        for index, layer in enumerate(self.layers):
            if self.args is not None:
                if self.args.use_pos_embed:
                    output = layer(self.with_pos_embed(output, pos)) + output
                else:
                    output = layer(output) + output
            

        if self.norm is not None:
            output = self.norm(output)

        return output 


class Decoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, args=None):
        super().__init__()
        self.args=args
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for index, layer in enumerate(self.layers):
            if self.args is not None:
                if self.args.use_pos_embed:
                    output = layer(self.with_pos_embed(output, pos), self.with_pos_embed(memory, query_pos)) + output
                else:
                    output = layer(output,memory) + output
           
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0) 


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Mamba(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
