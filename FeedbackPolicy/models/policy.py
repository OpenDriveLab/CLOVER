# import open_clip

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops import repeat

from .transformer_utils import Block, PatchEmbed, get_2D_position_embeddings, RMSNorm, SwishGLU


        

class MAPAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int) -> None:
        """Multi-Input Multi-Headed Attention Operation"""
        super().__init__()
        assert embed_dim % n_heads == 0, "`embed_dim` must be divisible by `n_heads`!"
        self.n_heads, self.scale = n_heads, (embed_dim // n_heads) ** -0.5

        # Projections (no bias) --> separate for Q (seed vector), and KV ("pool" inputs)
        self.q, self.kv = nn.Linear(embed_dim, embed_dim, bias=False), nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, seed: torch.Tensor, x: torch.Tensor, attention_mask = None) -> torch.Tensor:
        (B_s, K, C_s), (B_x, N, C_x) = seed.shape, x.shape
        assert C_s == C_x, "Seed vectors and pool inputs must have the same embedding dimensionality!"

        # Project Seed Vectors to `queries`
        q = self.q(seed).reshape(B_s, K, self.n_heads, C_s // self.n_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B_x, N, 2, self.n_heads, C_x // self.n_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # Attention --> compute weighted sum over values!
        scores = q @ (k.transpose(-2, -1) * self.scale)
        if attention_mask is not None:
            attention_mask = (
                attention_mask[:, None, None, :].repeat(1, self.n_heads, 1, 1) #.flatten(0, 1)
            )
            scores.masked_fill_(attention_mask == 0, float("-inf"))
        attn = scores.softmax(dim=-1)
        vals = (attn @ v).transpose(1, 2).reshape(B_s, K, C_s)

        # Project back to `embed_dim`
        return self.proj(vals)


### ======Token Aggregator===== ###
class TokenAggregation(nn.Module):
    def __init__(
        self,
        n_latents: int,
        embed_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        do_rms_norm: bool = True,
        do_swish_glu: bool = True,
        #add_internal_latents: bool = False,
    ) -> None:
        """Multiheaded Attention Pooling Block -- note that for MAP, we adopt earlier post-norm conventions."""
        super().__init__()
        self.n_latents, self.embed_dim, self.n_heads = n_latents, embed_dim, 2 * n_heads

        # Projection Operator
        self.projection = nn.Linear(embed_dim, self.embed_dim)

        # Custom MAP Attention (seed, encoder outputs) -> seed
        self.attn_norm = RMSNorm(self.embed_dim) if do_rms_norm else nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.attn = MAPAttention(self.embed_dim, n_heads=self.n_heads)

        # Position-wise Feed-Forward Components
        self.mlp_norm = RMSNorm(self.embed_dim) if do_rms_norm else nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            # Handle SwishGLU vs. GELU MLP...
            (
                SwishGLU(self.embed_dim, int(mlp_ratio * self.embed_dim))
                if do_swish_glu
                else nn.Sequential(nn.Linear(self.embed_dim, int(mlp_ratio * self.embed_dim)), nn.GELU())
            ),
            nn.Linear(int(mlp_ratio * self.embed_dim), self.embed_dim),
        )


    def forward(self, x: torch.Tensor, latents: torch.Tensor = None, mask = None) -> torch.Tensor:
        if len(latents.shape) == 2:
            latents = repeat(latents, "n_latents d -> bsz n_latents d", bsz=x.shape[0])

        latents = self.attn_norm(latents + self.attn(latents, self.projection(x), mask))
        latents = self.mlp_norm(latents + self.mlp(latents))
        return latents.squeeze(dim=1)


### ======Feature Fusion===== ###
class ConvFuser(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_num: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_num = patch_num
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.channel_selection = nn.Sequential(
                            nn.Linear(out_channels, out_channels),
                            nn.Sigmoid()
                            )

    def forward(self, inputs_rgb: torch.Tensor, inputs_depth: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([inputs_rgb, inputs_depth], dim=-1)
        inputs = rearrange(inputs, 'b (h w) d -> b d h w', h=self.patch_num, w=self.patch_num)
        feature = self.conv(inputs)
        feature = rearrange(feature, 'b d h w -> b (h w) d')
        selection_weights = self.channel_selection(feature.mean(dim=1))

        # channel-wise multiply
        feature = feature * selection_weights.unsqueeze(1)

        return feature



### ======Action Decoder===== ###
class MLPActionVelocityHead_Tanh(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Create a linear layer for each action
        self.num_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 6),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.num_head(x)
        return x


class MLPActionGripperHead(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Create a linear layer for each action

        self.bin_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.bin_head(x)
        return x



class FeedbackDrivenPolicy(nn.Module):
    def __init__(self, vision_encoder, vis_dim, window_size, sampling_step):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.window_size = window_size // sampling_step

        self.action_embed_cur = nn.Parameter(torch.zeros(1, vis_dim), requires_grad=True)
        nn.init.normal_(self.action_embed_cur, std=0.02)
        self.action_embed_tgt = nn.Parameter(torch.zeros(1, vis_dim), requires_grad=True)
        nn.init.normal_(self.action_embed_tgt, std=0.02)

        # ViT-S as depth encoder
        depth_res = 168
        patch_size = 14
        depth_dim = 384
        depth_encoder_layers = 6

        self.depth_patch2embed = PatchEmbed(
            resolution = depth_res, patch_size=patch_size, embed_dim=depth_dim, in_channels=1
        )
        self.depth_encoder_pe = nn.Parameter(
            torch.zeros(1, self.depth_patch2embed.num_patches, depth_dim),
            requires_grad=False,
        )
        enc_pe = get_2D_position_embeddings(
            depth_dim, int(self.depth_patch2embed.num_patches**0.5)
        )
        self.depth_encoder_pe.data.copy_(torch.from_numpy(enc_pe).float().unsqueeze(0))

        self.depth_encoder_blocks = nn.ModuleList(
            [
                Block(
                    embed_dim = depth_dim,
                    n_heads = 6,
                    mlp_ratio = 4,
                    do_rms_norm=True,
                    do_swish_glu=True,
                    do_layer_scale=True,
                )
                for _ in range(depth_encoder_layers)
            ]
        )

        # Multimodal feature fusion
        self.fuser = ConvFuser(in_channels=vis_dim + depth_dim, out_channels=vis_dim, patch_num = depth_res // patch_size)

        # Token aggregator
        self.token_aggregation = TokenAggregation(n_latents = 1, embed_dim = vis_dim, n_heads = 8)

        # Action decoder
        self.velo_head = MLPActionVelocityHead_Tanh(hidden_size = vis_dim)
        self.gripper_head = MLPActionGripperHead(hidden_size = vis_dim)
        

    def _encode_vision(self, vision_x: torch.Tensor):
        """
        Encode RGB inputs with VC-1.
        Args:
            vision_x (torch.Tensor): Vision input
        """
        b, T = vision_x.shape[:2]
        vision_x = rearrange(vision_x, "b T c h w -> (b T) c h w")

        with torch.no_grad():
            vision_x = self.vision_encoder(vision_x)

        vision_x = rearrange(vision_x, "(b T) d h w  -> b T (h w) d", b=b, T=T)
        return vision_x


    def _encode_vision_depth(self, vision_depth: torch.Tensor, state_tensor=None):
        """
        Encode depth map with ViT-S.
        Args:
            vision_depth (torch.Tensor): Depth map input
        """
        b, T = vision_depth.shape[:2]
        vision_depth = rearrange(vision_depth, "b T c h w -> (b T) c h w")
        patch_depth = self.depth_patch2embed(vision_depth) + self.depth_encoder_pe

        for block in self.depth_encoder_blocks:
            patch_depth = block(patch_depth)

        patch_depth = rearrange(patch_depth, "(b T) v d -> b T v d", b=b, T=T)
        return patch_depth


    def get_pred_features(self, vision_x, vision_depth):

        vision_rgb = self._encode_vision(vision_x)
        vision_depth = self._encode_vision_depth(vision_depth)


        fused_feature = []
        for i in range(vision_depth.shape[1]):
            fused_feature.append(self.fuser(vision_rgb[:,i], vision_depth[:,i]))
        fused_feature = torch.stack(fused_feature, dim=1)

        aggregated_tgt = []

        for i in range(fused_feature.shape[1]):
            aggregated_tgt.append(self.token_aggregation(fused_feature[:,i], self.action_embed_tgt))
        
        return aggregated_tgt


    def forward(self, vision_x, vision_depth):
        

        ### Multimodal Encoder
        vision_rgb = self._encode_vision(vision_x)
        vision_depth = self._encode_vision_depth(vision_depth)

        fused_feature = []
        for i in range(vision_rgb.shape[1]):
            fused_feature.append(self.fuser(vision_rgb[:,i], vision_depth[:,i]))
        fused_feature = torch.stack(fused_feature, dim=1)


        ### Token Aggregator
        aggregated_tgt = self.token_aggregation(fused_feature[:,-1], self.action_embed_tgt)

        stacked_velo_pred = []
        stacked_grip_pred = []
        state_estimation = []
        is_close_pred = []

        for i in range(fused_feature.shape[1] - 1):
            aggregated_cur = self.token_aggregation(fused_feature[:,i], self.action_embed_cur)
            
            ### Error measurement
            cos_distance = 1 - F.cosine_similarity(F.normalize(aggregated_tgt), F.normalize(aggregated_cur))
            aggregated_tokens = aggregated_tgt - aggregated_cur

            ### Decode Actions
            velo_pred = self.velo_head(aggregated_tokens)
            grip_pred = self.gripper_head(aggregated_tokens)

            stacked_velo_pred.append(velo_pred)
            stacked_grip_pred.append(grip_pred)

        
        stacked_velo_pred = torch.stack(stacked_velo_pred, dim=1)
        stacked_grip_pred = torch.stack(stacked_grip_pred, dim=1)


        return stacked_velo_pred, stacked_grip_pred, cos_distance


