# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from Networks.LEM import LocalityEnhanceModule
from Networks.CrossAttnDecoder import CrossAttnBlock


class VisionTransformer_token(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        trunc_normal_(self.pos_embed, std=.02)

        self.output1 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 1)
        )
        self.output1.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        x = self.output1(x)

        return x


class VisionTransformer_gap(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        trunc_normal_(self.pos_embed, std=.02)

        self.output1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(6912 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        self.output1.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        x = x[:, 1:]

        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        x = F.adaptive_avg_pool1d(x, (48))
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.output1(x)
        return x


# This is a implementation of TransCrowd+Decoder.
class VisionTransformer_gap_decoder(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.uni_embed = nn.Parameter(torch.zeros(1, 256, 1024))
        self.dec_depth = 4
        self.patch_size = 16
        hidden_num = 768
        self.input_size = 48
        self.output_size = 48
        self.inner_index, self.outer_index = self.get_index()
        # initialize the weight of decoder, psm and qem
        self.qem = LocalityEnhanceModule(hidden_num=hidden_num, input_size=384, outout_size=384,
                                         patch_size=self.patch_size)
        self.transformer_decoder = nn.ModuleList([
            CrossAttnBlock(
                dim=hidden_num, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                init_values=0., window_size=None)
            for _ in range(self.dec_depth)])
        trunc_normal_(self.pos_embed, std=.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.output1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(6912 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        self.output2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(10 * 768, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
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
        B = x.shape[0]
        x = self.patch_embed(x)

        ori_cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((ori_cls_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x[:, 1:])

        query_embed = self.qem(x)

        cls_tokens = self.cls_token.expand(B, 10, -1)
        tgt_outer = query_embed
        tgt_outer = torch.cat((cls_tokens, tgt_outer), dim=1)
        for i, dec in enumerate(self.transformer_decoder):
            tgt_outer = dec(tgt_outer, x)
        tgt_outer = self.norm(tgt_outer)
        return tgt_outer[:, 10:], tgt_outer[:, 0:10]

    def forward(self, x):
        x, y = self.forward_features(x)
        x = F.adaptive_avg_pool1d(x, (48))
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        x = self.output1(x)
        y = self.output2(y)
        return x, y


@register_model
def base_patch16_384_token(pretrained=True, **kwargs):
    model = VisionTransformer_token(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        '''download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth'''
        checkpoint = torch.load(
            './Networks/deit_base_patch16_384-8de9b5d1.pth')
        model.load_state_dict(checkpoint["model"], strict=False)
        print("load transformer pretrained")
    return model


@register_model
def base_patch16_384_gap(pretrained=True, **kwargs):
    model = VisionTransformer_gap(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        '''download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth'''
        checkpoint = torch.load(
            './Networks/deit_base_patch16_384-8de9b5d1.pth')
        model.load_state_dict(checkpoint["model"], strict=False)
        print("load transformer pretrained")
    return model


@register_model
def base_patch16_384_gap_decoder(pretrained=True, **kwargs):
    model = VisionTransformer_gap_decoder(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        '''download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth'''
        checkpoint = torch.load(
            './Networks/deit_base_patch16_384-8de9b5d1.pth')
        model.load_state_dict(checkpoint["model"], strict=False)
        print("load transformer pretrained")
    return model
