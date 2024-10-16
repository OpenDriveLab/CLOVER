from .diffusion_model.unet import UNetModel
from torch import nn
import torch
from einops import repeat, rearrange


class VisualPlanner(nn.Module):
    # ResidualBlocks + ExtendedSelfAttn + CalsualTemporalAttn
    def __init__(self, 
                image_size, 
                in_channels, 
                out_channels,                   # output channels (RGB + Depth)
                dims=3,                         # dimension of conv blocks
                temporal_length=8,              # num of video frames
                use_vae=False,                  # whether to use VAE for frame encoding / decoding
                decoupled_output=False,         # decoupled RGB & Depth output
                decoupled_input=False,          # decoupled RGB & Depth input
                flow_reg=False, 
                with_state_estimate=False
    ):
        super(VisualPlanner, self).__init__()
        self.unet = UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=64,
            out_channels=out_channels,
            num_res_blocks=2,
            attention_resolutions=(8, 16, ),
            dropout=0,
            channel_mult=(1, 2, 3, 4, 5),
            conv_resample=True,
            dims=dims,                      # whether to use temporal 1D-Conv
            num_classes=None,
            task_tokens=True,
            task_token_channels=768,        # 768 for CLIP-Large Text Encoder
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
            decoupled_output=decoupled_output,
            decoupled_input=decoupled_input,
            temporal_length=temporal_length,
            simple_adapter=False,
            flow_reg=flow_reg,
        )
        self.use_vae = use_vae
        self.dims = dims
        self.flow_reg = flow_reg


    def forward(self, x, t, text_embedding=None, x_cond = None, **kwargs):
        if x_cond is not None:
            x_cond = repeat(x_cond.squeeze(1), 'b c h w -> b f c h w', f=x.shape[1])
            x = torch.cat([x_cond, x], dim=2) # ( b, f, c * 2 , H, W )
        
        b, f, c, h, w = x.shape
        if self.dims == 2:
            x = rearrange(x, 'b f c h w -> (b f) c h w')
        else:
            x = x.transpose(1,2)    # ( b, c * 2 , f, H, W )
        
        if self.flow_reg and kwargs['forward']:
                out, flow = self.unet(x, t, text_embedding, **kwargs)
        else:
            out = self.unet(x, t, text_embedding, **kwargs)

        if self.dims == 2:
            out = rearrange(out, '(b f) c h w -> b f c h w', b=b)
        else:
            out = out.transpose(1,2)

        if self.flow_reg and kwargs['forward']:
                return out, flow
        else:
            return out



    

