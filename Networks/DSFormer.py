import math
import numpy as np
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.registry import register_model
from einops import rearrange, repeat
# from timm.models import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from Networks.LEM import LocalityEnhanceModule
from Networks.LEM import LocalityEnhanceModule4VanilaViT
from Networks.LEM import ASPPBlock
from Networks.SwinTransformerStable import SwinTransformerStable
from timm.models.vision_transformer import VisionTransformer, _cfg
from Networks.CrossAttnDecoder import CrossAttnBlock
import torch
import torch.nn as nn

from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class DensityTokenNet(SwinTransformerStable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.uni_embed = nn.Parameter(torch.zeros(1, 256, 1024))
        self.dec_depth = 4
        hidden_num = 1024
        self.input_size = 48
        self.output_size = 48
        self.inner_index, self.outer_index = self.get_index()
        # initialize the weight of decoder, psm and lem
        self.qem = LocalityEnhanceModule(hidden_num=hidden_num, input_size=48, outout_size=48,
                                        patch_size=self.patch_size)
        self.transformer_decoder = nn.ModuleList([
            CrossAttnBlock(
                dim=hidden_num, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                init_values=0., window_size=None)
            for _ in range(self.dec_depth)])
        trunc_normal_(self.pos_embed, std=.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1024))
        self.output1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(6912, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        self.output2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(10 * 1024, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
        self.output1.apply(self._init_weights)
        self.output2.apply(self._init_weights)

    def get_index(self):
        input_query_width = self.input_size // self.patch_size
        output_query_width = self.output_size // self.patch_size
        mask = torch.ones(size=[output_query_width, output_query_width]).long()
        pad_width = (output_query_width - input_query_width) // 2
        mask[pad_width:-pad_width, pad_width:-pad_width] = 0
        mask = mask.view(-1)
        return mask == 0, mask == 1

    def forward_features(self, x):
        batch_size = x.shape[0]
        num_cls_tokens = 10
        cls_tokens = self.cls_token.expand(batch_size, num_cls_tokens, -1)
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        # x = self.norm(x)
        query_embed = self.qem(x)
        tgt_outer = query_embed
        tgt_outer = torch.cat((cls_tokens, tgt_outer), dim=1)
        for i, dec in enumerate(self.transformer_decoder):
            tgt_outer = dec(tgt_outer, x)
        tgt_outer = self.norm(tgt_outer)
        return tgt_outer[:, num_cls_tokens:], tgt_outer[:, 0:num_cls_tokens]


    def forward(self, x):
        x,y = self.forward_features(x)
        x = F.adaptive_avg_pool1d(x, 48)
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        x = self.output1(x)
        y = self.output2(y)
        return x,y


@register_model
def model_DSFormer(pretrained=False, **kwargs, ):
    model = DensityTokenNet(
        img_size=384, patch_size=4, in_chans=3,
        embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
        window_size=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            './pretrained_weights/swin_base_patch4_window12_384.pth')
        state_dict = checkpoint['model']
        print("load transformer pretrained")
        model.load_state_dict(state_dict, strict=False)
    return model

