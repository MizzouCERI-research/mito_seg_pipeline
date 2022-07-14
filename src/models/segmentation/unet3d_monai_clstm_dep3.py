
# DERIVED MODEL: CONSTRUCTOR

# %%
import os
import sys
import shutil
import glob
import logging
import numpy as np
np.random.seed(100)
import matplotlib.pyplot as plt

import re
import scipy.stats
import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 0)
pd.set_option('display.max_columns', 0)

import SimpleITK as sitk

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    from torchinfo import summary
except: 
    from torchsummary import summary
import segmentation_models_pytorch as smp
import monai
from monai.data import Dataset
from monai.utils import get_seed, progress_bar
from monai.transforms import \
    apply_transform, Compose, Randomizable, Transform, \
    ToTensord, AddChanneld, ScaleIntensityd, ScaleIntensityRanged, \
    Resized, CropForegroundd, RandCropByPosNegLabeld, RandSpatialCropd, \
    RandRotate90d, Orientationd, RandAffined, Spacingd

torch.cuda.empty_cache()


# %%

# from typing import Optional, Union, List
# # from src.smp.unet.decoder import UnetDecoder
# from src.smp.unet.decoder3d import UnetDecoder3D
# from src.smp.unet.decoder3d_clstm import UnetDecoder3DCLstm
# from src.smp.encoders import get_encoder
# from src.smp.base import SegmentationModel
# from src.smp.base.heads import SegmentationHead, ClassificationHead
# from src.smp.base.heads import SegmentationHead3D, ClassificationHead3D

# # import src.monai.nets.densenet as densenet
# import src.monai.nets as nets
# from src.smp.encoders.encoder_monai import EncoderMonaiCLstm


# %%
from typing import Optional, Union, List

# from src.models.segmentation.encoder import get_encoder
from src.models.segmentation.encoder.encoder_monai import EncoderMonaiCLstm

# from src.models.segmentation.decoder.decoder3d import UnetDecoder3D
from src.models.segmentation.decoder.decoder3d_clstm import UnetDecoder3DCLstm

from src.models.segmentation.model import SegmentationModel
from src.models.segmentation.heads import SegmentationHead3D, ClassificationHead3D

import monai.networks.nets as nets


# %%
class Unet3D_CLSTM(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_depth (int): number of stages used in decoder, larger depth - more features are generated.
            e.g. for depth=3 encoder will generate list of features with following spatial shapes
            [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature tensor will have
            spatial resolution (H/(2^depth), W/(2^depth)]
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
            One of [True, False, 'inplace']
        decoder_attention_type: attention module used in decoder of the model
            One of [``None``, ``scse``]
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function to apply after final convolution;
            One of [``sigmoid``, ``softmax``, ``logsoftmax``, ``identity``, callable, None]
        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)

    Returns:
        ``torch.nn.Module``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """
    # # from src.smp.encoders.general_resnet_clstm import GeneralEncoderCLstm
    # from src.smp.encoders.general_resnet_clstm import GeneralEncoderCLstm
    # # from torchsummary import summary
    # class CustomEncoder(GeneralEncoderCLstm):
    #     def __init__(self, spatial_dims, in_channels, block_config, out_channels, depth,  **kwargs) -> None:

    #         self.backbone_name = 'resnet'
    #         self.layer_list = None
    #         # self.ignore_list = ['transition']
    #         super().__init__(self.backbone_name, self.layer_list, **kwargs)
    #         # super().__init__(**kwargs)
            
            

    #         self.spatial_dims = spatial_dims
    #         self.block_config = block_config
    #         self.classify_classes = 2

    #         self._out_channels = out_channels # all out channels
    #         self._depth = depth
    #         self._in_channels = in_channels

    #         # self.model = densenet.densenet121(spatial_dims=self.spatial_dims, in_channels=self._in_channels, out_channels=self.classify_classes)
    #         self.model = nets.se_resnext50_32x4d(spatial_dims=self.spatial_dims, in_channels=self._in_channels, num_classes=self.classify_classes, pretrained = False, progress = True)
    #         # summary(self.model, input_size=(1, 64, 64, 64))

    #         #  These two last layers are not included in the layers of function "get_stages()", not necessarily delete
    #         del self.model.last_linear
    #         del self.model.adaptive_avg_pool

           

        
        # def get_stages(self):
        #     return []


        


    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        num_clstm_layers = 1,
        device = torch.device("cpu"), # all devices
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()



        # import monai
        # from torchsummary import summary
        # import segmentation_models_pytorch as smp
        
        self.spatial_dims = 3
        self.in_channel = 1
        self.encoder_block_config = []
        # self.encoder_out_channels = (1, 64, 128, 256, 512)
        # self.decoder_channels = (128, 64, 32, 16)
        # self. encoder_depth = 4
        self.encoder_out_channels = (1, 32, 64, 128, 256)
        self.decoder_channels = (128, 64, 32)
        self. encoder_depth = encoder_depth

        

        self.decoder_use_batchnorm = True
        self.decoder_attention_type = None
        self.encoder_name = 'monai_resnet'
        self.segment_classes = classes
        self.num_clstm_layers = num_clstm_layers
        self.device = device

        # self.encoder = get_encoder(
        #     encoder_name,
        #     in_channels=in_channels,
        #     depth=encoder_depth,
        #     weights=encoder_weights,
        # )
        


        # self.encoder = densenet.densenet121(spatial_dims=3, in_channels=1, out_channels=2)
        # del self.encoder.class_layers
        # self.encoder.class_layers = None
        # self.encoder = self.CustomEncoder(spatial_dims=self.spatial_dims, in_channels=self.in_channel,
                                        # block_config=self.encoder_block_config, out_channels=self.encoder_out_channels, 
                                        # depth=self. encoder_depth)

        self.encoder = EncoderMonaiCLstm('EncoderMonaiClstm',
                                    dimensions=3,
                                    in_channels=1,
                                    out_channels=3,
                                    channels=(32, 64, 128, 256),
                                    strides=(2, 2, 2),
                                    num_res_units=16,
                                    norm='INSTANCE',
                                    ratio=0.25,
                                )

        # self.decoder = UnetDecoder(
        #     encoder_channels=self.encoder.out_channels,
        #     decoder_channels=decoder_channels,
        #     n_blocks=encoder_depth,
        #     use_batchnorm=decoder_use_batchnorm,
        #     center=True if encoder_name.startswith("vgg") else False,
        #     attention_type=decoder_attention_type,
        # )

        self.decoder = UnetDecoder3DCLstm(
            encoder_channels=self.encoder_out_channels,
            decoder_channels=self.decoder_channels,
            n_blocks=self.encoder_depth,
            use_batchnorm=self.decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=self.decoder_attention_type,
            num_classes=self.segment_classes,
            num_clstm_layers = self.num_clstm_layers,
            device = self.device,
            )

        
        # self.segmentation_head = SegmentationHead(
        #     in_channels=decoder_channels[-1],
        #     out_channels=classes,
        #     activation=activation,
        #     kernel_size=3,
        # )

        self.segmentation_head = SegmentationHead3D(
            # in_channels=self.decoder_channels[-1],
            in_channels=64, # by checking decoder output
            out_channels=self.segment_classes,
            activation=activation,
            kernel_size=3,
            upsampling=2.0,
        )

        # if aux_params is not None:
        #     self.classification_head = ClassificationHead(
        #         in_channels=self.encoder.out_channels[-1], **aux_params
        #     )
        # else:
        #     self.classification_head = None

        if aux_params is not None:
            self.classification_head = ClassificationHead3D(
            in_channels=self.encoder.out_channels[-1], **aux_params
        )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()









# # %%
# # -----------------------------------------------------------
# # TENSORBOARD
# # (GENERATING GRAPH ONLY WORKS WITH CPU)

# model_u3 = Unet3D_CLSTM(encoder_name='seresnet', encoder_depth=3, encoder_weights=None, 
#                 decoder_channels=(64, 32, 16), in_channels=1, classes=1, num_clstm_layers = 1, device = torch.device("cpu"),
#                 activation=None, decoder_attention_type=None)


# # %%
# print(model_u3)


# # %%
# inputs = torch.rand(1, 1, 64, 64, 64)
# with SummaryWriter(comment='my_unet3d-resnext_50_4-clstm', log_dir='runs/my_unet3d-densenet121-clstm') as w:
#     w.add_graph(model_u3, (inputs, ))
# w.close()










# # %%
# # -----------------------------------------------------------
# # SUMMARY


# # %%
# model_u3 = Unet3D_CLSTM(encoder_name='seresnet', encoder_depth=3, encoder_weights=None, 
#                 decoder_channels=(64, 32, 16), in_channels=1, classes=1, num_clstm_layers = 1, device = torch.device("cuda"),
#                 activation=None, decoder_attention_type=None)

# # %%
# print(model_u3)


# # %%
# inputs = torch.rand(1, 1, 64, 64, 64).cuda()
# model_u3 = model_u3.cuda()
# model_u3.eval()
# outputs = model_u3(inputs)


# # %%
# # model_u3 = model_u3.cuda()
# # summary(model_u3, input_size=(1, 512, 512))
# # summary(model_u3, input_size=(1, 192, 192, 192))
# summary(model_u3, (1, 64, 64, 64), depth=12, col_names=["kernel_size", "output_size", "num_params", "mult_adds"],)

# # Depth = 4
# # ===========================================================================
# # Layer (type:depth-idx)                             Param #
# # ===========================================================================
# # ├─CustomEncoder: 1-1                               --
# # |    └─SENet: 2-1                                  --
# # |    |    └─Sequential: 3-1                        22,080
# # |    |    └─Sequential: 3-2                        258,864
# # |    |    └─Sequential: 3-3                        1,477,760
# # |    |    └─Sequential: 3-4                        8,700,288
# # |    |    └─Sequential: 3-5                        17,893,760
# # |    |    └─AdaptiveAvgPool3d: 3-6                 --
# # |    |    └─Linear: 3-7                            4,098
# # ├─UnetDecoder3DCLstm: 1-2                          --
# # |    └─ModuleList: 2-2                             --
# # |    |    └─ConvLSTM3D: 3-8                        905,977,856
# # |    |    └─ConvTranspose3d: 3-9                   33,556,480
# # |    |    └─ConvLSTM3D: 3-10                       679,485,440
# # |    |    └─ConvTranspose3d: 3-11                  16,778,240
# # |    |    └─ConvLSTM3D: 3-12                       169,873,408
# # |    |    └─ConvTranspose3d: 3-13                  4,194,816
# # |    |    └─ConvLSTM3D: 3-14                       42,469,376
# # |    |    └─ConvTranspose3d: 3-15                  1,048,832
# # ├─SegmentationHead3D: 1-3                          --
# # |    └─Conv3d: 2-3                                 6,913
# # |    └─Upsample: 2-4                               --
# # |    └─Activation: 2-5                             --
# # |    |    └─Identity: 3-16                         --
# # ===========================================================================
# # Total params: 1,881,748,211
# # Trainable params: 1,881,748,211
# # Non-trainable params: 0
# # ===========================================================================


# # %%
# # Densenet
# # summary(model_u3, input_size=(1, 192, 192, 192))

# # Depth = 3
# # =================================================================
# # Layer (type:depth-idx)                   Param #
# # =================================================================
# # ├─CustomEncoder: 1-1                     --
# # |    └─DenseNet: 2-1                     --
# # |    |    └─Sequential: 3-1              11,242,624
# # ├─UnetDecoder3DCLstm: 1-2                --
# # |    └─ModuleList: 2-2                   --
# # |    |    └─ConvLSTM3D: 3-2              226,496,512
# # |    |    └─ConvTranspose3d: 3-3         8,389,632
# # |    |    └─ConvLSTM3D: 3-4              169,873,408
# # |    |    └─ConvTranspose3d: 3-5         4,194,816
# # |    |    └─ConvLSTM3D: 3-6              42,469,376
# # |    |    └─ConvTranspose3d: 3-7         1,048,832
# # ├─SegmentationHead3D: 1-3                --
# # |    └─Conv3d: 2-3                       6,913
# # |    └─Upsample: 2-4                     --
# # |    └─Activation: 2-5                   --
# # |    |    └─Identity: 3-8                --
# # =================================================================
# # Total params: 463,722,113
# # Trainable params: 463,722,113
# # Non-trainable params: 0
# # =================================================================



# # %%
# print(model_u3)


# # %%
# model_u3.decoder.clstm



# # %%
# model_u3.decoder.conv_trans


# # %%
# for sublayer in model_u3.decoder.conv_trans:
#     print(sublayer)