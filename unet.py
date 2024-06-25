# from typing import List, Optional, Union, Any
# import segmentation_models_pytorch as smp
import torch
from monai.networks import nets
from monai.networks.layers import Norm

# class UNet(smp.Unet):
#     def __init__(
#         self,
#         encoder_name: str = "resnet34",
#         # encoder_depth: int = 5,
#         # encoder_weights: Union[str] = "imagenet",
#         # decoder_use_batchnorm: bool = True,
#         # decoder_channels: List[int] = (256, 128, 64, 32, 16),
#         # decoder_attention_type: Optional[str] = None,
#         in_channels: int = 1,
#         classes: int = 5,
#         # activation: Union[str, Any] = None,
#         # aux_params: Optional[dict] = None,
#     ):
#         super().__init__(
#             encoder_name,
#             # encoder_depth,
#             # encoder_weights,
#             # decoder_use_batchnorm,
#             # decoder_channels,
#             # decoder_attention_type,
#             in_channels,
#             classes,
#             # activation,
#             # aux_params,
#         )
    
#     def forward(self, **kwargs) -> torch.Tensor:
#         pixel_values = kwargs["image"]
#         return super().forward(pixel_values)
class UNet(nets.UNet):
    def __init__(self,spatial_dims, in_channels, out_channels, 
                 channels,strides):
        super().__init__(spatial_dims, in_channels, out_channels, channels, strides)

    def forward(self, **kwargs) -> torch.Tensor:
        image = kwargs["image"]
        return super().forward(image)
        
