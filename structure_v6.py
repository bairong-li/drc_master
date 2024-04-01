# !/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2022 Charles
'''
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import re
import cv2
from thop import profile
import torch


def lut_transform(imgs, luts):
    # img (b, 3, h, w), lut (b, c, m, m, m)

    # normalize pixel values
    imgs = (imgs - .5) * 2.
    # reshape img to grid of shape (b, 1, h, w, 3)
    grids = imgs.permute(0, 2, 3, 1).unsqueeze(1)

    # after gridsampling, output is of shape (b, c, 1, h, w)
    outs = F.grid_sample(luts, grids,
                         mode='bilinear', padding_mode='border', align_corners=True)
    # remove the extra dimension
    outs = outs.squeeze(2)

    return outs


def srgb2lrgb_t(I0, gvalue=2.4):
    gamma = ((I0 + 0.055) / 1.055) ** gvalue
    scale = I0 / 12.92
    return torch.where(I0 > 0.04045, gamma, scale)


def srgb2lrgb_n(I0, gvalue=2.4):
    gamma = ((I0 + 0.055) / 1.055) ** gvalue
    scale = I0 / 12.92
    return np.where(I0 > 0.04045, gamma, scale)


class encoder(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super(encoder, self).__init__()
        self.conv = nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=2, padding=1, padding_mode='replicate')
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class decoder4(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, H=256, W=256):
        super(decoder4, self).__init__()
        self.conv = nn.Conv2d(
            in_nc, out_nc, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.resize = nn.Upsample(size=(H, W), mode='bilinear')

    def forward(self, x):
        y = self.act(self.conv(x))
        z = self.resize(y)
        return z


class ClassifierLTM4(nn.Module):
    def __init__(self, n_ranks, backbone_para):
        super(ClassifierLTM4, self).__init__()
        chn_in = 3
        height, width = backbone_para['down_resolution']
        img_height, img_width = backbone_para['org_resolution']
        block_overlap = backbone_para['block_overlap']

        self.enresize1 = nn.Identity()

        stride_h = img_height // height
        stride_w = img_width // width
        assert img_height % height == 0, f"{img_height}%{height}!=0"
        assert img_width % width == 0, f"{img_width}%{width}!=0"
        kernel_size_h = stride_h + 2 * block_overlap
        kernel_size_w = stride_w + 2 * block_overlap
        self.enresize2 = nn.AvgPool2d(kernel_size=(kernel_size_h, kernel_size_w), stride=(stride_h, stride_w), padding=block_overlap)

        self.encoder2 = encoder(chn_in, 32)  # 128*128
        self.encoder4 = encoder(32, 64)  # 64 *64
        self.encoder8 = encoder(64, 128)  # 32 *32
        self.encoder16 = encoder(128, 256)  # 16 *16
        self.encoder32 = encoder(256, 512)  # 8  *8

        self.decoder16 = decoder4(512 + 512, 256, height // 16, width // 16)  # 16 *16
        self.decoder8 = decoder4(256 + 256, 128, height // 8, width // 8)  # 32 *32
        self.decoder4 = decoder4(128 + 128, 64, height // 4, width // 4)  # 64 *64
        self.decoder2 = decoder4(64 + 64, 32, height // 2, width // 2)  # 128*128
        self.decoder1 = decoder4(32 + 32, 16, height // 1, width // 1)  # 256*256

        self.conv = nn.Conv2d(16, n_ranks, kernel_size=3, stride=1, padding=1)
        # self.deresize = nn.Upsample(size=(H, W),mode='bilinear')

        self.icls = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.LayerNorm((1, 1)),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.LayerNorm((1, 1)),
            nn.Conv2d(128, n_ranks, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):

        enresize = self.enresize2(self.enresize1(x))

        # print(enresize.shape)
        # a = self.encoder2.conv.weight.T
        encoder2 = self.encoder2(enresize)
        encoder4 = self.encoder4(encoder2)
        encoder8 = self.encoder8(encoder4)
        encoder16 = self.encoder16(encoder8)
        encoder32 = self.encoder32(encoder16)

        encoderout = torch.mean(encoder32, dim=[2, 3], keepdim=True)
        # decoderin = torch.tile(encoderout, (1, 1, encoder32.shape[-2], encoder32.shape[-1]))
        decoderin = encoderout.repeat(1, 1, encoder32.shape[-2], encoder32.shape[-1])

        decoder16 = self.decoder16(torch.cat([decoderin, encoder32], dim=1))

        decoder8 = self.decoder8(torch.cat([decoder16, encoder16], dim=1))
        decoder4 = self.decoder4(torch.cat([decoder8, encoder8], dim=1))
        decoder2 = self.decoder2(torch.cat([decoder4, encoder4], dim=1))
        decoder1 = self.decoder1(torch.cat([decoder2, encoder2], dim=1))

        icls = self.icls(encoderout)
        # pcls = self.deresize(self.conv(decoder1))

        _, _, H, W = x.shape
        # print(H, W)
        deresize = nn.Upsample(size=(H, W), mode='bilinear')
        pcls = deresize(self.conv(decoder1))
        # print(pcls.shape)

        return pcls, icls


TC_weiht = np.array([[[[0.0625, 0.1875, 0.1875, 0.0625],
                        [0.1875, 0.5625, 0.5625, 0.1875],
                        [0.1875, 0.5625, 0.5625, 0.1875],
                        [0.0625, 0.1875, 0.1875, 0.0625]]]]).astype(np.float32)
TC_weiht = torch.from_numpy(TC_weiht)
class FreTransferInv(nn.Module):
    def __init__(self, chn, ks=4):
        super(FreTransferInv, self).__init__()
        self.net1 = nn.ConvTranspose2d(chn, chn, kernel_size=(ks, ks), stride=(ks//2, ks//2), padding=ks-1-(ks//2-1)//2, output_padding=0, bias=None, groups=chn)  # Cin = 1, Cout = 4, kernel_size = (1,2)
        self.net1.weight = torch.nn.Parameter(TC_weiht.repeat(chn, 1, 1, 1), requires_grad=False)
        self.pad = nn.ReplicationPad2d(1) 
    def forward(self, x):
        y = self.pad(x)
        out = self.net1(y)
        return out


class decoder5(nn.Module):
    def __init__(self, in_nc=3, out_nc=3):
        super(decoder5, self).__init__()
        self.conv = nn.Conv2d(
            in_nc, out_nc, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.resize = FreTransferInv(chn=out_nc)
    def forward(self, x):
        y = self.act(self.conv(x))
        out = self.resize(y)
        return out


class ClassifierLTM5(nn.Module):
    def __init__(self, n_ranks, backbone_para):
        super(ClassifierLTM5, self).__init__()
        chn_in = 3
        height, width = backbone_para['down_resolution']
        img_height, img_width = backbone_para['org_resolution']
        block_overlap = backbone_para['block_overlap']

        self.enresize1 = nn.Identity()

        stride_h = img_height // height
        stride_w = img_width // width
        assert img_height % height == 0, f"{img_height}%{height}!=0"
        assert img_width % width == 0, f"{img_width}%{width}!=0"
        kernel_size_h = stride_h + 2 * block_overlap
        kernel_size_w = stride_w + 2 * block_overlap
        self.enresize2 = nn.AvgPool2d(kernel_size=(kernel_size_h, kernel_size_w), stride=(stride_h, stride_w), padding=block_overlap)

        self.encoder2 = encoder(chn_in, 32)  # 128*128
        self.encoder4 = encoder(32, 64)  # 64 *64
        self.encoder8 = encoder(64, 128)  # 32 *32
        self.encoder16 = encoder(128, 256)  # 16 *16
        self.encoder32 = encoder(256, 512)  # 8  *8

        self.decoder16 = decoder5(512 + 512, 256)  # 16 *16
        self.decoder8 = decoder5(256 + 256, 128)  # 32 *32
        self.decoder4 = decoder5(128 + 128, 64)  # 64 *64
        self.decoder2 = decoder5(64 + 64, 32)  # 128*128
        self.decoder1 = decoder5(32 + 32, 16)  # 256*256

        self.conv = nn.Conv2d(16, n_ranks, kernel_size=3, stride=1, padding=1)
        # self.deresize = nn.Upsample(size=(H, W),mode='bilinear')

        self.icls = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, n_ranks, kernel_size=1, stride=1, padding=0)
        )

        assert img_height // height == img_width // width, 'not equal scale'
        assert int(np.log2(img_height // height))-np.log2(img_height // height)==0, 'not power of 2'
        num = int(np.log2(img_height // height))
        models = []
        for _ in range(num):
            models.append(FreTransferInv(chn=n_ranks))
        self.deresize = nn.Sequential(*models)

    def forward(self, x):

        enresize = self.enresize2(self.enresize1(x))

        # print(enresize.shape)
        # a = self.encoder2.conv.weight.T
        encoder2 = self.encoder2(enresize)
        encoder4 = self.encoder4(encoder2)
        encoder8 = self.encoder8(encoder4)
        encoder16 = self.encoder16(encoder8)
        encoder32 = self.encoder32(encoder16)

        encoderout = torch.mean(encoder32, dim=[2, 3], keepdim=True)
        # decoderin = torch.tile(encoderout, (1, 1, encoder32.shape[-2], encoder32.shape[-1]))
        decoderin = encoderout.repeat(1, 1, encoder32.shape[-2], encoder32.shape[-1])

        decoder16 = self.decoder16(torch.cat([decoderin, encoder32], dim=1))

        decoder8 = self.decoder8(torch.cat([decoder16, encoder16], dim=1))
        decoder4 = self.decoder4(torch.cat([decoder8, encoder8], dim=1))
        decoder2 = self.decoder2(torch.cat([decoder4, encoder4], dim=1))
        decoder1 = self.decoder1(torch.cat([decoder2, encoder2], dim=1))

        icls = self.icls(encoderout)
        decoder = self.conv(decoder1)
        decoder = decoder*icls
        weight = self.deresize(decoder)

        _, _, H, W = x.shape
        deresize = nn.Upsample(size=(H, W), mode='bilinear')
        weight1 = deresize(self.conv(decoder1))*icls
        weight2 = deresize(self.conv(decoder1)*icls)
        d = torch.mean(torch.abs(weight1-weight2))
        print(d)

        return weight


class SA_LUT3DGenerator(nn.Module):
    def __init__(self, n_colors, n_vertices, n_ranks, weight_softmax, backbone_para) -> None:
        super().__init__()
        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_ranks = n_ranks
        self.weight_softmax = weight_softmax
        self.basis_luts_bank = nn.Linear(n_ranks, n_colors * (n_vertices ** n_colors), bias=False)

    def forward(self, weights, img):

        weights = weights.unsqueeze(2)  # (b, 1, n, h, w)
        if self.weight_softmax:
            weights = F.softmax(weights, dim=2)

        basis_luts = self.basis_luts_bank.weight.t().view(self.n_ranks, self.n_colors,
                                                          *((self.n_vertices,) * self.n_colors))
        basis_luts = basis_luts.unsqueeze(1)
        basis_luts = basis_luts.repeat(1, img.shape[0], 1, 1, 1, 1)
        outs = []
        for i in range(basis_luts.shape[0]):
            outs.append(
                lut_transform(img, basis_luts[i])
            )
        outs = torch.stack(outs, dim=0).permute(1, 0, 2, 3, 4)  # b, n, c, h, w

        outs = torch.mul(weights, outs)
        outs = torch.sum(outs, dim=1, keepdim=False)

        return outs


class DRC_model(nn.Module):
    def __init__(self, n_colors, n_vertices, n_ranks, weight_softmax, backbone_para) -> None:
        super().__init__()
        backbone_type = backbone_para['backbone_type']
        if backbone_type=='LTM4':
            self.backbone = ClassifierLTM4(n_ranks, backbone_para)
        elif backbone_type=='LTM5':
            self.backbone = ClassifierLTM5(n_ranks, backbone_para)
        self.lut3d_generator = SA_LUT3DGenerator(n_colors, n_vertices, n_ranks, weight_softmax, backbone_para)
        self.loadpretrained(
            f'{path}/2024-02-19-11_51_41_seplut_hdrplus-dvp_unet5_fix3d_simple_average_256_0_trans0.6-0.2-0.7_unet4/iter_135200.pth')

    def forward(self, img):
        weights = self.backbone(img)
        outs = self.lut3d_generator(weights, img)
        return outs

    def loadpretrained(self, model_para):
        weights_dict = torch.load(model_para)['state_dict']
        revise_keys = [(r'^module\.', '')]
        metadata = getattr(weights_dict, '_metadata', OrderedDict())
        for p, r in revise_keys:
            state_dict_backbone = OrderedDict(
                {re.sub(p, r, '.'.join(k.split('.')[1:])): v
                 for k, v in weights_dict.items() if 'backbone' in k and 'gbackbone' not in k})
        state_dict_backbone._metadata = metadata

        missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict_backbone, strict=False)
        if len(unexpected_keys) != 0 or len(missing_keys) != 0:
            for missing_key in missing_keys:
                missing_key = missing_key.split('.')
                if not ('decoder' in missing_key[0] and 'resize'==missing_key[1] and 'net1'==missing_key[2]):
                    if not ('deresize' in missing_key[0]):
                        raise f'backbone: len(unexpected_keys)={len(unexpected_keys)}, len(missing_keys)={len(missing_keys)}'

        for p, r in revise_keys:
            state_dict_lut3d = OrderedDict(
                {re.sub(p, r, '.'.join(k.split('.')[1:])): v
                 for k, v in weights_dict.items() if 'lut3d_generator' in k})
        state_dict_lut3d._metadata = metadata
        missing_keys, unexpected_keys = self.lut3d_generator.load_state_dict(state_dict_lut3d, strict=False)
        if len(unexpected_keys) != 0 or len(missing_keys) != 0:
            raise f'lut3d_generator: len(unexpected_keys)={len(unexpected_keys)}, len(missing_keys)={len(missing_keys)}'
    
path = '/home/bairong.li/240106/DRC_Deploy'
bit = 8
gvalue = 1.8
backbone_type = 'LTM5'


if __name__ == '__main__':

    if bit == 8:
        backbone_para = dict(
            backbone_type=backbone_type,
            up_resolution=None,
            org_resolution=(1536, 2816), # 1520 padding to 1536, 2688 padding to 2816
            down_resolution=(192, 352),
            block_overlap=0,
        )
    else:
        backbone_para = dict(
            backbone_type=backbone_type,
            up_resolution=None,
            org_resolution=(1536, 2560), # 1440 padding to 1536
            down_resolution=(192, 320),
            block_overlap=0,
        )

    net2 = DRC_model(n_colors=3, n_vertices=17, n_ranks=5, weight_softmax=False, backbone_para=backbone_para)
    model = net2.cuda().eval()

    with open(f'{path}/dataset/test-{bit}bit.txt', 'r') as f:
        ffiles = f.readlines()
        for fname in ffiles:
            fname_ = fname.strip()[:-4]
            extension = fname.strip()[-3:]
            img_in = cv2.imread(f'{path}/dataset/test-{bit}bit/{fname_}.{extension}', cv2.IMREAD_UNCHANGED)
            img_in = cv2.cvtColor(img_in, cv2.COLOR_RGB2BGR)
            if bit == 8:
                img_in = np.concatenate((np.repeat(img_in[:, :1, :], 64, axis=1), img_in, np.repeat(img_in[:, -1:, :], 64, axis=1)), axis=1)
                img_in = np.concatenate((np.repeat(img_in[:1, :, :], 8, axis=0), img_in, np.repeat(img_in[-1:, :, :], 8, axis=0)), axis=0)
                img_in = img_in / 255.0
                img_in = srgb2lrgb_n(img_in, gvalue)
            else:
                img_in = np.concatenate((cv2.flip(img_in[:48, :, :], 0), img_in, cv2.flip(img_in[-48:, :, :], 0)), axis=0)
                img_in = img_in / 65535.0
            img_in = img_in.astype(np.float32)
            img_in = torch.from_numpy(img_in).unsqueeze(0).permute((0, 3, 1, 2)).cuda()

            img_out2 = model(img_in)
            img_np2 = img_out2.detach().cpu().squeeze().permute((1, 2, 0)).numpy()
            img_np2 = (np.clip(img_np2, 0, 1) * 255.0).round()
            img_np2 = img_np2.astype(np.uint8)
            img_np2 = cv2.cvtColor(img_np2, cv2.COLOR_RGB2BGR)
            if bit == 8:
                img_np2 = img_np2[8:-8, 64:-64, :]
            else:
                img_np2 = img_np2[48:-48, :, :]
            cv2.imwrite(f'{path}/output2/{fname_}' + f'_{backbone_type}.png', img_np2)