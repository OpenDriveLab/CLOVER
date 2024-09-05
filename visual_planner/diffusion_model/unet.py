from abc import abstractmethod

import math

import numpy as np
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F

from einops import rearrange, repeat

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

from rotary_embedding_torch import (
            RotaryEmbedding,
            apply_rotary_emb
        )

from .imagen import PerceiverResampler

from .diffusion_utils import get_1d_sincos_temp_embed

from raft_utils.raft import RAFT


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)

class FinalLayer(nn.Module):
    """
    Output layer with FiLM conditioning mechanism
    """
    def __init__(self, dims, input_ch, out_channels, time_embed_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(input_ch, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(input_ch, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * input_ch, bias=True)
        )

        # Zero-out output layers:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, emb):
        b,c,f, *spatial = x.shape
        x = rearrange(x, 'b c f x y -> b f (x y) c', f=f, c=c)
        emb = emb.unsqueeze(1).repeat(1,f,1)
        shift, scale = self.adaLN_modulation(emb).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        x = rearrange(x, 'b f (x y) c -> b c f x y', f=f, x=spatial[0], y=spatial[1])
        return x



class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TransformerBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, context = None):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context = None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, TransformerBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class CrossAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None, RoPE=False):
        super(CrossAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)


    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        """_summary_

        Args:
            v (_type_): bs, n_img, dim
            l (_type_): bs, n_text, dim
            attention_mask_v (_type_, optional): _description_. bs, n_img
            attention_mask_l (_type_, optional): _description_. bs, n_text

        Returns:
            _type_: _description_
        """
        bsz, tgt_len, _ = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self.l_proj(l)

        key_states = self._shape(key_states, -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)

        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # bs*nhead, nimg, ntxt

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(
                attn_weights, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(
                attn_weights, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        # mask language for vision
        if attention_mask_l is not None:
            attention_mask_l = (
                attention_mask_l[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights.masked_fill_(attention_mask_l == 0, float("-inf"))
        attn_weights_v = attn_weights.softmax(dim=-1)


        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_output_v = torch.bmm(attn_probs_v, value_l_states)


        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)

        return attn_output_v


# Uni-Direction MHA (text->image, image->text)
class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        v_dim,
        l_dim,
        embed_dim,
        num_heads,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
        cfg=None,
        RoPE=False,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(CrossAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = CrossAttention(
            v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, RoPE=RoPE
        )

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)


    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v = self.attn(
            v, l, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l
        )
        v = v + self.drop_path(self.gamma_v * delta_v)

        return v




class ExtendedSpatialAttention(TransformerBlock):
    """
    An attention block that allows spatial positions to attend to each other.
    We "extand" the original spatial attention, now the k-th frame can also attend to (k-1)-th frame.
    This is to encourage timeporal consistency.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        lang_dim = 512,
        dims=2,
        temporal_length=8,
        RoPE=True,
        spatial_ds=1,
    ):
        super().__init__()
        self.channels = channels
        self.dims = dims
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.norm_out = normalization(channels)

        self.temporal_length = temporal_length
        self.extended_self_attn = CrossAttentionBlock(channels, channels, channels, num_heads=num_heads)
        self.window_mask = torch.tril(torch.ones(temporal_length,temporal_length),diagonal= 0) - \
                           torch.tril(torch.ones(temporal_length,temporal_length),diagonal=-2)

        self.cross_attn = CrossAttentionBlock(channels, lang_dim, channels, num_heads=num_heads)

        self.RoPE = RoPE
        spatial_ds = 1
        if self.RoPE:
            self.pos_emb = RotaryEmbedding(
                dim = 16,
                freqs_for = 'pixel',
                max_freq = 256,
                learned_freq = True,
            )

        

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context,), self.parameters(), True)

    def _forward(self, x, context=None):

        if self.dims == 2:
            bs, c, *spatial = x.shape
            x = rearrange(x, "(b f) c x y -> b c f x y", c=c, f=self.temporal_length)
        
        b, c, f, *spatial = x.shape
        if self.RoPE:
            freqs = self.pos_emb.get_axial_freqs(f, spatial[0], spatial[1])
            x = rearrange(x, "b c f x y -> b f x y c")
            x = apply_rotary_emb(freqs, x)  # rotate in frequencies
            x = rearrange(x, "b f x y c -> (b f) c (x y)")
        else:
            x = rearrange(x, "b c f x y -> (b f) c (x y)")

        x = self.norm(x)
        x = rearrange(x, "(b f) c (x y) -> b f (x y) c", f = f, x = spatial[0], y = spatial[1])

        x_all = []
        for idx, mask in enumerate(self.window_mask):
            attn_idx = torch.nonzero(mask)
            x_attn = rearrange(x[:,attn_idx].squeeze(2), "b f l d -> b (l f) d")
            x_f = self.extended_self_attn(x[:,idx], x_attn)
            x_all.append(x_f)

        # BUG: Residual connections already in the attention module
        # We keep it here for reproduction
        x = torch.stack(x_all, dim=1) + x   

        x = rearrange(x, "b f (x y) c -> (b f) c (x y)", x = spatial[0], y = spatial[1])
        x = self.norm_out(x).transpose(1,2)

        x = self.cross_attn(x, context.repeat(f, 1, 1))

        if self.dims == 2:
            return rearrange(x, "b (x y) c -> b c x y", c=c, x=spatial[0], y=spatial[1])
        else:
            return rearrange(x, "(b f) (x y) c -> b c f x y", c=c, f=f, x=spatial[0], y=spatial[1])


class RelativePosition(nn.Module):
    """
    https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py
    """

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings


class TemporalAttention(TransformerBlock):
    """
    Causal Temporal Attention with relative position embedding
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_relative_position=True,
        temporal_length=8,
        dropout=0.1,
        dims=2,
        with_norm=False,
    ):
        super().__init__()
        dim_head = channels // num_heads

        self.dims = dims
        self.scale = dim_head ** -0.5
        self.heads = num_heads
        self.temporal_length = temporal_length
        self.use_relative_position = use_relative_position
        self.with_norm = with_norm
        if self.with_norm:
            self.norm = nn.LayerNorm(channels)
            self.norm_out = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(channels, channels, bias=False)
        self.to_v = nn.Linear(channels, channels, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Dropout(dropout)
        )

        if use_relative_position:
            assert temporal_length is not None
            self.relative_position_k = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
            self.relative_position_v = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)

        self.causual_mask = torch.tril(torch.ones(temporal_length, temporal_length)).view(1,temporal_length,temporal_length)



    def forward(self, x, context=None):
        # reshape for temporal modeling
        if self.dims == 2:
            bs, c, *spatial = x.shape
            x = rearrange(x, "(b f) c x y -> (b x y) f c", c=c, f=self.temporal_length, x=spatial[0], y=spatial[1])
        else:
            bs, c, f, *spatial = x.shape
            x = rearrange(x, "b c f x y -> (b x y) f c")
        # x = rearrange(x, "(b t) hw c -> (b hw) t c", t=self.temporal_length)
        if self.with_norm:
            x = self.norm(x)
        num_heads = self.heads

        # calculate qkv
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=num_heads), (q, k, v))
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        # relative positional embedding
        if self.use_relative_position:
            len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
            k2 = self.relative_position_k(len_q, len_k)
            sim2 = einsum("b t d, t s d -> b t s", q, k2) * self.scale
            sim += sim2

        # mask attention
        _MASKING_VALUE = -1e+16 if sim.dtype == torch.float32 else -1e+4
        sim = sim.masked_fill(self.causual_mask.to(sim.device) == 0, _MASKING_VALUE)

        # attend to values
        attn = sim.softmax(dim=-1)
        out = einsum("b i j, b j d -> b i d", attn, v)

        # relative positional embedding
        if self.use_relative_position:
            v2 = self.relative_position_v(len_q, len_v)
            out2 = einsum("b t s, t s d -> b t d", attn, v2)
            out += out2

        # merge attention heads
        out = rearrange(out, "(b h) n d -> b n (h d)", h=num_heads)
        out = self.to_out(out) + x

        if self.with_norm:
            out = self.norm_out(out)

        # reshape back
        if self.dims == 2:
            out = rearrange(out, "(b x y) f c -> (b f) c x y", f=self.temporal_length, c=c, x=spatial[0], y=spatial[1])
        else:
            out = rearrange(out, "(b x y) f c -> b c f x y", f=f, c=c, x=spatial[0], y=spatial[1])
        return out



class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        task_tokens=True,
        task_token_channels=512,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        decoupled_output=False,
        decoupled_input=False,
        temporal_length=8,
        simple_adapter=False,
        flow_reg=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.task_tokens = task_tokens
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.decoupled_output = decoupled_output
        self.temporal_length = temporal_length
        self.simple_adapter = simple_adapter
        self.flow_reg = flow_reg



        # Optical-flow Regularization: we only incorporate an additional context model
        # For further details, plz refer to: https://arxiv.org/pdf/2003.12039
        if self.flow_reg:
            self.optical_module = RAFT(small=True)


        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.dims = dims
        if dims == 2:
            self.temp_embed = nn.Parameter(torch.zeros(1, temporal_length, time_embed_dim), requires_grad=False)
            temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
            self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        if task_tokens:
            if simple_adapter:
                self.task_attnpool = nn.Sequential(
                    linear(task_token_channels, time_embed_dim),
                    nn.SiLU(),
                    linear(time_embed_dim, time_embed_dim),
            )
            else:
                self.task_attnpool = nn.Sequential(
                    PerceiverResampler(dim=task_token_channels, depth=2, num_latents=32, max_seq_len=32),
                    nn.Linear(task_token_channels, time_embed_dim),
                )
        ch = input_ch = int(channel_mult[0] * model_channels)

        self.decoupled_input = decoupled_input
        if self.decoupled_input:
            self.input_emebd = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, 6, ch // 2, 3, padding=1)),
             TimestepEmbedSequential(conv_nd(dims, 2, ch // 2, 3, padding=1))]
        )
            self.input_blocks = nn.ModuleList([nn.Identity()])
        else:
            self.input_blocks = nn.ModuleList(
                [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
            )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        ExtendedSpatialAttention(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            lang_dim = time_embed_dim,
                            dims=dims,
                            temporal_length=temporal_length,
                            spatial_ds=ds
                        )
                    )
                    layers.append(
                        TemporalAttention(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            temporal_length=temporal_length,
                            dims=dims,
                            with_norm=True,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            ExtendedSpatialAttention(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            lang_dim = time_embed_dim,
                            dims=dims,
                            temporal_length=temporal_length,
                            spatial_ds=ds
                        ),
            TemporalAttention(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            temporal_length=temporal_length,
                            dims=dims,
                            with_norm=True,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        ExtendedSpatialAttention(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            lang_dim = time_embed_dim,
                            dims=dims,
                            temporal_length=temporal_length,
                            spatial_ds=ds
                    ))
                    layers.append(
                        TemporalAttention(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            temporal_length=temporal_length,
                            dims=dims,
                            with_norm=True,
                    ))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        if self.decoupled_output:
            self.out = nn.ModuleList(
            [   nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                conv_nd(dims, input_ch, 3, 3, padding=1),
            ),
                nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                conv_nd(dims, input_ch, 1, 3, padding=1),
            )]
            )
        else:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                conv_nd(dims, input_ch, out_channels, 3, padding=1),
            )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None, **kwargs):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None or self.task_tokens
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        
        if self.task_tokens:
            context = self.task_attnpool(y)#.mean(dim=1)
            emb = emb + context.mean(dim=1)

        if self.dims == 2:
            emb = repeat(emb, 'b c -> (b f) c', f = self.temporal_length)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)

        optical_flow_pred = []
        for idx, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

            # We conduct optical flow computation only with 1 / 16 features
            if idx == 4 and kwargs['forward'] and self.flow_reg:
                x_start = kwargs['x_start'][:,:,:3]     # RGB Only
                for i in range(self.temporal_length - 1):
                    optical_flow_pred.append(self.optical_module(x_start[:,i], h[:,:,i], h[:,:,i+1])) # (bs, xy, h, w)
        h = h.type(x.dtype)


        if kwargs['forward'] and self.flow_reg:
            return self.out(h), optical_flow_pred
        else:
            return self.out(h)


