import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d, deform_conv2d


class DeformConv(DeformConv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=None):
        super(DeformConv, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        channels_ = groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(input, offset, self.weight, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation, mask=mask)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ResidualBlock(nn.Module):
    def __init__(self, planes, ):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, 3, 1, 1)
        self.conv2 = DeformConv(planes, planes, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(planes, affine=True)
        self.norm2 = nn.InstanceNorm2d(planes, affine=True)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x_sc = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + x_sc


class LocalityEnhanceModule(nn.Module):
    def __init__(self, hidden_num=768, n_block=8, input_size=128, outout_size=192, patch_size=16, embed_dim=128):
        super(LocalityEnhanceModule, self).__init__()

        self.hidden_num = hidden_num
        self.input_query_width = input_size // patch_size
        self.output_query_width = outout_size // patch_size
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_num) for _ in range(n_block)])
        self.norm = nn.LayerNorm(hidden_num)
        self.embed = nn.Linear(hidden_num, hidden_num)
        self.inner_query_index, self.outer_query_index = self.get_index()

    def get_index(self):
        mask = torch.ones(size=[self.output_query_width, self.output_query_width]).long()
        pad_width = (self.output_query_width - self.input_query_width) // 2
        mask[pad_width:-pad_width, pad_width:-pad_width] = 0
        mask = mask.view(-1)
        return mask == 0, mask == 1

    def forward(self, src_query):
        b, n, c = src_query.size()

        ori_src_query = src_query
        # assert src_query.size(
        #     1) == self.input_query_width ** 2, f'QEM input spatial dimension is wrong, {src_query.size(1)} and {self.input_query_width ** 2}'
        # noise = torch.randn(size=(b, self.output_query_width ** 2, c // 8), dtype=torch.float32).to(src_query.device)

        x = src_query.permute(0, 2, 1).reshape(b, c, self.input_query_width, self.input_query_width).contiguous()

        for layer in self.res_blocks:
            x = layer(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        # x[:, self.inner_query_index, :] = ori_src_query
        x = self.embed(x)
        return x


class LocalityEnhanceModule4VanilaViT(nn.Module):
    def __init__(self, hidden_num=768, n_block=8, input_size=128, outout_size=192, patch_size=16, embed_dim=128):
        super(LocalityEnhanceModule4VanilaViT, self).__init__()

        self.hidden_num = hidden_num
        self.input_query_width = input_size // patch_size
        self.output_query_width = outout_size // patch_size
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_num) for _ in range(n_block)])
        self.norm = nn.LayerNorm(hidden_num)
        self.embed = nn.Linear(hidden_num, hidden_num)
        self.inner_query_index, self.outer_query_index = self.get_index()

    def get_index(self):
        mask = torch.ones(size=[self.output_query_width, self.output_query_width]).long()
        pad_width = (self.output_query_width - self.input_query_width) // 2
        mask[pad_width:-pad_width, pad_width:-pad_width] = 0
        mask = mask.view(-1)
        return mask == 0, mask == 1

    def forward(self, src_query):
        b, n, c = src_query.size()

        temp = src_query[:, 0, :].unsqueeze(1)

        x = src_query[:, 1:, :].permute(0, 2, 1).reshape(b, c, self.input_query_width,
                                                         self.input_query_width).contiguous()

        for layer in self.res_blocks:
            x = layer(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((temp, x), dim=1)
        x = self.norm(x)

        x = self.embed(x)
        return x

class ASPPBlock(nn.Module):
    def __init__(self, hidden_num=768, n_block=8, input_size=128, outout_size=192, patch_size=16, embed_dim=128):
        super(ASPPBlock, self).__init__()

        self.hidden_num = hidden_num
        self.input_query_width = input_size // patch_size
        self.output_query_width = outout_size // patch_size
        self.blocks = nn.ModuleList([ASPP(in_channels=1024,atrous_rates=[6,12,18],out_channels=1024) for _ in range(n_block)])
        self.norm = nn.LayerNorm(hidden_num)
        self.embed = nn.Linear(hidden_num, hidden_num)
        self.inner_query_index, self.outer_query_index = self.get_index()

    def get_index(self):
        mask = torch.ones(size=[self.output_query_width, self.output_query_width]).long()
        pad_width = (self.output_query_width - self.input_query_width) // 2
        mask[pad_width:-pad_width, pad_width:-pad_width] = 0
        mask = mask.view(-1)
        return mask == 0, mask == 1

    def forward(self, src_query):
        b, n, c = src_query.size()
        x = src_query.permute(0, 2, 1).reshape(b, c, self.input_query_width, self.input_query_width).contiguous()
        for layer in self.blocks:
            x = layer(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.embed(x)
        return x

if __name__ == '__main__':
    m1 = LocalityEnhanceModule()
    x1 = torch.randn([1, 64, 768])
    y1 = m1(x1)
    print(y1.size())
