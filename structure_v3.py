# !/usr/bin/env
# -*- coding: utf-8 -*-

'''
Copyright (c) 2022 Charles
'''
CUDA_VISIBLE_DEVICES=3
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import re
import cv2

rec_filt_w = np.array([1, 1, 1, 1]) / 4.
rec_filt_w = torch.from_numpy(rec_filt_w)
class FreTransferInv(nn.Module):
    def __init__(self, inc, outc, ):
        super(FreTransferInv, self).__init__()
        self.net1 = nn.ConvTranspose2d(inc, outc, kernel_size=(2, 2), stride=(2, 2), padding=0, bias=None, groups=1)  # Cin = 1, Cout = 4, kernel_size = (1,2)
        self.net1.weight = torch.nn.Parameter(rec_filt_w, requires_grad=False)  # torch.Size([2, 1, 1, 2])

    def forward(self, x):
        out = self.net1(x)
        return out
    
    
def ini_3dlut(identity_lut1, n_ranks, n_colors, n_vertices):
	coordinate_range = range(n_vertices-1)
	all_coordinates = []
	for x in coordinate_range:
		for y in coordinate_range:
			for z in coordinate_range:
				neigh_coordinates = []
				for i in [0, 1]:
					for j in [0, 1]:
						for k in [0, 1]:
							new_x, new_y, new_z = x + i, y + j, z + k
							neigh_coordinates.append([new_x, new_y, new_z])

				all_coordinates.append(neigh_coordinates)

	luts_idx = np.vstack(all_coordinates)
	identity_lut2 = identity_lut1[:, :, luts_idx[:,0], luts_idx[:,1], luts_idx[:,2]].reshape(n_ranks*n_colors, -1, 8).permute(2, 1, 0)
	return identity_lut2


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


def lut_transform2(grids, luts):
	# grids (b, c, h, w); lut (n, c, m, m, m)
	grids = grids.permute((1, 0, 2, 3)).unsqueeze(0) # (1, c, b, h, w)
	ldim = luts.size()[-1] - 1
	grids_xyz = torch.floor(grids * ldim) # 3, h, w
	dxyz = grids * ldim - grids_xyz # 3, h, w

	grids_xyz = grids_xyz.type(torch.long) # 3, h, w
	r, g, b = 2, 1, 0 #012, 021, 102, 120, 210, 201
	ridx = grids_xyz[0, r, :, :, :]
	gidx = grids_xyz[0, g, :, :, :]
	bidx = grids_xyz[0, b, :, :, :]
	# rdis = dxyz[:, r:r+1, :, :, :]
	# gdis = dxyz[:, g:g+1, :, :, :]
	# bdis = dxyz[:, b:b+1, :, :, :]

	dxyz = dxyz.permute((0, 3, 2, 1, 4))
	rdis = dxyz[:, :, :, r:r+1, :].repeat((5, 1, 1, 3, 1))
	gdis = dxyz[:, :, :, g:g+1, :].repeat((5, 1, 1, 3, 1))
	bdis = dxyz[:, :, :, b:b+1, :].repeat((5, 1, 1, 3, 1))

	m_rdis = 1 - rdis
	m_gdis = 1 - gdis
	m_bdis = 1 - bdis
	a_ridx = ridx+1
	a_bidx = bidx+1
	a_gidx = gidx+1

	m_rdis_m_gdis = m_rdis*m_gdis
	rdis_m_gdis = rdis*m_gdis
	gdis_m_bdis = gdis*m_bdis
	gdis_bdis = gdis*  bdis

	outs = \
	m_rdis_m_gdis*m_bdis*luts[:, :, ridx  , gidx  , bidx  ].permute((0, 3, 2, 1, 4)) + \
	m_rdis*gdis_m_bdis*luts[:, :, ridx  , a_gidx, bidx  ].permute((0, 3, 2, 1, 4)) + \
	m_rdis_m_gdis*bdis*luts[:, :, ridx  , gidx  , a_bidx].permute((0, 3, 2, 1, 4)) + \
	  m_rdis*gdis_bdis*luts[:, :, ridx  , a_gidx, a_bidx].permute((0, 3, 2, 1, 4)) + \
	  rdis_m_gdis*m_bdis*luts[:, :, a_ridx, gidx  , bidx  ].permute((0, 3, 2, 1, 4)) + \
	  rdis_m_gdis*bdis*luts[:, :, a_ridx, gidx  , a_bidx].permute((0, 3, 2, 1, 4)) + \
	  rdis*gdis_m_bdis*luts[:, :, a_ridx, a_gidx, bidx  ].permute((0, 3, 2, 1, 4)) + \
		rdis*gdis_bdis*luts[:, :, a_ridx, a_gidx, a_bidx].permute((0, 3, 2, 1, 4))

	outs = outs.permute((0, 3, 2, 1, 4))
	return outs

def lut_transform3(grids, luts, n, m):
	# grids (b, c, h, w); lut (8, m*m*m, n*c)
	r, g, b = 2, 1, 0
	bsize, _, hsize, wsize = grids.shape
	grids = grids.permute((1, 0, 2, 3)) # (c, b, h, w)
	grids_xyz = torch.floor(grids * m).type(torch.long)

	# coordinates
	idx_r = grids_xyz[r, :, :, :]	   # (b, h, w)
	idx_g = grids_xyz[g, :, :, :]
	idx_b = grids_xyz[b, :, :, :]
	idx_rgb = (idx_r * m * m + idx_g * m + idx_b)

	# coefficients
	dis_xyz = grids * m - grids_xyz	 # (3, b, h, w)
	dis_r = dis_xyz[r, :, :, :]		 # (b, h, w)
	dis_g = dis_xyz[g, :, :, :]
	dis_b = dis_xyz[b, :, :, :]
	dis_mr = 1 - dis_r
	dis_mg = 1 - dis_g
	dis_mb = 1 - dis_b
 
	# results
	luts_val = luts[:, idx_rgb, :]	  # (8, b, h, w, n*c)
	outs = \
	(dis_mr * dis_mg * dis_mb).unsqueeze(-1) * luts_val[0] + \
	(dis_mr * dis_mg * dis_b ).unsqueeze(-1) * luts_val[1] + \
	(dis_mr * dis_g  * dis_mb).unsqueeze(-1) * luts_val[2] + \
	(dis_mr * dis_g  * dis_b ).unsqueeze(-1) * luts_val[3] + \
	(dis_r  * dis_mg * dis_mb).unsqueeze(-1) * luts_val[4] + \
	(dis_r  * dis_mg * dis_b ).unsqueeze(-1) * luts_val[5] + \
	(dis_r  * dis_g  * dis_mb).unsqueeze(-1) * luts_val[6] + \
	(dis_r  * dis_g  * dis_b ).unsqueeze(-1) * luts_val[7]
	outs = outs.reshape(bsize, hsize, wsize, n, -1)

	return outs


class encoder(nn.Module):
	def __init__(self, in_nc=3, out_nc=3):
		super(encoder, self).__init__()
		self.conv = nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=2, padding=1)
		self.act = nn.LeakyReLU(0.2, inplace=True)

	def forward(self, x):
		return self.act(self.conv(x))


class decoder4(nn.Module):
	def __init__(self, in_nc=3, out_nc=3, H=256, W=256):
		super(decoder4, self).__init__()
		self.conv = nn.Conv2d(
			in_nc, out_nc, kernel_size=3, stride=1, padding=1)
		self.act = nn.LeakyReLU(0.2, inplace=True)
		self.resize = nn.Upsample(size=(H, W), mode='bilinear')

	def forward(self, x):
		y = self.act(self.conv(x))
		z = self.resize(y)
		return z

class decoder5(nn.Module):
	def __init__(self, in_nc=3, out_nc=3, H=256, W=256):
		super(decoder5, self).__init__()
		self.conv = nn.Conv2d(
			in_nc, out_nc, kernel_size=3, stride=1, padding=1)
		self.act = nn.LeakyReLU(0.2, inplace=True)
		self.resize = FreTransferInv(inc=in_nc, outc=out_nc)

	def forward(self, x):
		y = self.act(self.conv(x))
		z = self.resize(y)
		return z

class ClassifierLTM4(nn.Module):
	def __init__(self, n_ranks, backbone_para):
		super(ClassifierLTM4, self).__init__()
		chn_in = 3
		self.out_channels = 512
		height, width = backbone_para['down_resolution']
		block_overlap = backbone_para['block_overlap']
		img_height = 1440
		img_width = 2560

		if backbone_para['up_resolution'] is not None:
			self.enresize1 = nn.Upsample(size=backbone_para['up_resolution'], mode='bilinear')
			img_height = backbone_para['up_resolution'][0]
			img_width = backbone_para['up_resolution'][1]
		else:
			self.enresize1 = nn.Identity()

		stride_h = img_height//height
		stride_w = img_width//width
		assert img_height%height==0, f"1440%{height}!=0"
		assert img_width%width==0, f"2560%{width}!=0"
		kernel_size_h = stride_h+2*block_overlap
		kernel_size_w = stride_w+2*block_overlap
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


class ClassifierLTM5(nn.Module):
	def __init__(self, n_ranks, backbone_para):
		super(ClassifierLTM5, self).__init__()
		chn_in = 3
		self.out_channels = 512
		height, width = backbone_para['down_resolution']
		block_overlap = backbone_para['block_overlap']
		img_height = 1440
		img_width = 2560

		if backbone_para['up_resolution'] is not None:
			self.enresize1 = nn.Upsample(size=backbone_para['up_resolution'], mode='bilinear')
			img_height = backbone_para['up_resolution'][0]
			img_width = backbone_para['up_resolution'][1]
		else:
			self.enresize1 = nn.Identity()

		stride_h = img_height//height
		stride_w = img_width//width
		assert img_height%height==0, f"1440%{height}!=0"
		assert img_width%width==0, f"2560%{width}!=0"
		kernel_size_h = stride_h+2*block_overlap
		kernel_size_w = stride_w+2*block_overlap
		self.enresize2 = nn.AvgPool2d(kernel_size=(kernel_size_h, kernel_size_w), stride=(stride_h, stride_w), padding=block_overlap)
		
		self.encoder2 = encoder(chn_in, 32)  # 128*128
		self.encoder4 = encoder(32, 64)  # 64 *64
		self.encoder8 = encoder(64, 128)  # 32 *32
		self.encoder16 = encoder(128, 256)  # 16 *16
		self.encoder32 = encoder(256, 512)  # 8  *8

		self.decoder16 = decoder5(512 + 512, 256, height // 16, width // 16)  # 16 *16
		self.decoder8 = decoder5(256 + 256, 128, height // 8, width // 8)  # 32 *32
		self.decoder4 = decoder5(128 + 128, 64, height // 4, width // 4)  # 64 *64
		self.decoder2 = decoder5(64 + 64, 32, height // 2, width // 2)  # 128*128
		self.decoder1 = decoder5(32 + 32, 16, height // 1, width // 1)  # 256*256

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



class SA_LUT3DGenerator(nn.Module):
	def __init__(self, n_colors, n_vertices, n_ranks, weight_softmax, backbone_para) -> None:
		super().__init__()
		self.n_colors = n_colors
		self.n_vertices = n_vertices
		self.n_ranks = n_ranks
		self.weight_softmax = weight_softmax
		self.basis_luts_bank = nn.Linear(n_ranks, n_colors * (n_vertices ** n_colors), bias=False)
		self.basis_luts_bank2 = 0


	def forward(self, ilut3d_weights, plut3d_weights, img):

		weights = ilut3d_weights * plut3d_weights
		weights = weights.unsqueeze(2)				# (b, 1, n, h, w)
		if self.weight_softmax:
			weights = F.softmax(weights, dim=2)
	
		basis_luts = self.basis_luts_bank.weight.t().view(self.n_ranks, self.n_colors, *((self.n_vertices,) * self.n_colors))
		basis_luts = basis_luts.unsqueeze(1)
		basis_luts = basis_luts.repeat(1, img.shape[0], 1, 1, 1, 1)
		outs = []
		for i in range(basis_luts.shape[0]):
			outs.append(
				lut_transform(img, basis_luts[i])
			)
		outs = torch.stack(outs, dim=0).permute(1, 0, 2, 3, 4) # b, n, c, h, w

		# outs3 = lut_transform3(img, self.basis_luts_bank2, self.n_ranks, self.n_vertices-1) # b, h, w, n, c
		# outs3 = outs3.permute((0, 3, 4, 1, 2))		  # (b, c, n, h, w)

		# basis_luts = self.basis_luts_bank.weight.t().view(self.n_ranks, self.n_colors, *((self.n_vertices,) * self.n_colors))
		# outs2 = lut_transform2(img, basis_luts) # c, b, n, 1440, 2560
		# outs2 = outs2.permute((2, 0, 1, 3, 4)) # b, c, n, 1440, 2560
  
		# print("\n==============\ndiff:",torch.mean(torch.abs(outs2-outs)))

		outs = torch.mul(weights, outs)
		outs = torch.sum(outs, dim=1, keepdim=False)

		return outs


class DRC_model(nn.Module):
	def __init__(self, n_colors, n_vertices, n_ranks, weight_softmax, backbone_para) -> None:
		super().__init__()
		self.backbone = ClassifierLTM4(n_ranks, backbone_para)
		self.lut3d_generator = SA_LUT3DGenerator(n_colors, n_vertices, n_ranks, weight_softmax, backbone_para)
		self.loadpretrained(f'/home/bairong.li/240106/DRC_Deploy/2024-02-19-11_51_41_seplut_hdrplus-dvp_unet5_fix3d_simple_average_256_0_trans0.6-0.2-0.7_unet4/iter_135200.pth')
		basis_luts2 = self.lut3d_generator.basis_luts_bank.weight.t().view(n_ranks, n_colors, *((n_vertices,) * n_colors))
		self.lut3d_generator.basis_luts_bank2 = ini_3dlut(basis_luts2, n_ranks, n_colors, n_vertices)


	def forward(self, img):
		ilut3d_weights, plut3d_weights = self.backbone(img)
		outs = self.lut3d_generator(ilut3d_weights, plut3d_weights, img)
		return outs

	 
	def loadpretrained(self, model_para):
		# basis_luts1 = self.lut3d_generator.basis_luts_bank.weight.t().view(n_ranks, n_colors, *((n_vertices,) * n_colors))
		weights_dict = torch.load(model_para)['state_dict']
		revise_keys = [(r'^module\.', '')]
		metadata = getattr(weights_dict, '_metadata', OrderedDict())
		for p, r in revise_keys:
			state_dict_backbone = OrderedDict(
				{re.sub(p, r, '.'.join(k.split('.')[1:])): v
				for k, v in weights_dict.items() if 'backbone' in k and 'gbackbone' not in k})
		state_dict_backbone._metadata = metadata
		# a = state_dict_backbone['encoder2.conv.weight']
		# b = self.backbone.encoder2.conv.weight.T.permute((3, 1, 2, 0))
		missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict_backbone, strict=True)
		if len(unexpected_keys) != 0 or len(missing_keys)!=0:
			raise f'backbone: len(unexpected_keys)={len(unexpected_keys)}, len(missing_keys)={len(missing_keys)}'
		# c = self.backbone.encoder2.conv.weight.T
		
		for p, r in revise_keys:
			state_dict_lut3d = OrderedDict(
				{re.sub(p, r, '.'.join(k.split('.')[1:])): v
				for k, v in weights_dict.items() if 'lut3d_generator' in k})
		state_dict_lut3d._metadata = metadata
		missing_keys, unexpected_keys = self.lut3d_generator.load_state_dict(state_dict_lut3d, strict=True)
		if len(unexpected_keys) != 0 or len(missing_keys)!=0:
			raise f'lut3d_generator: len(unexpected_keys)={len(unexpected_keys)}, len(missing_keys)={len(missing_keys)}'
		# basis_luts2 = self.lut3d_generator.basis_luts_bank.weight.t().view(n_ranks, n_colors, *((n_vertices,) * n_colors))
		# basis_luts3 = weights_dict['lut3d_generator.basis_luts_bank.weight'].t().view(n_ranks, n_colors, *((n_vertices,) * n_colors))
		print(1)
		

if __name__ == '__main__':

	n_colors=3
	n_vertices=17
	n_ranks=5
	# backbone_para = dict(
	# 	up_resolution=(2560, 2560),
	# 	down_resolution=(256, 256),
	# 	block_overlap=0,
	# )
	backbone_para = dict(
		up_resolution=None,
		down_resolution=(288, 256),
		block_overlap=0,
	)
	
	# net = DRC_model(n_colors, n_vertices, n_ranks, False, backbone_para=backbone_para)
	# model = net.cuda().eval()
 
	net2 = DRC_model(n_colors, n_vertices, n_ranks, False, backbone_para=backbone_para)
	model = net2.cuda().eval()

	with open('/home/bairong.li/240106/DRC_Deploy/dataset/test-16bit.txt', 'r') as f:
		ffiles = f.readlines()
	for fname in ffiles:
		fname_ = fname.strip()[:-4]
		img_in = cv2.imread(f'/home/bairong.li/240106/DRC_Deploy/dataset/test-16bit/{fname_}.ppm', cv2.IMREAD_UNCHANGED)
		img_in = cv2.cvtColor(img_in, cv2.COLOR_RGB2BGR)/65535.0
		img_in = img_in.astype(np.float32)
		img_in = torch.from_numpy(img_in).unsqueeze(0).permute((0, 3, 1, 2)).cuda()
		# img_in = np.load(f'/home/bairong.li/240106/DRC_Deploy/input/{fname_}_lq'+'.npy')
		# img_in = torch.from_numpy(img_in).unsqueeze(0).permute((0, 3, 1, 2)).cuda()
		
		img_out = model(img_in)
		img_np = img_out.detach().cpu().squeeze().permute((1, 2, 0)).numpy()
		np.save(f'/home/bairong.li/240106/DRC_Deploy/output2/{fname_}'+'.npy', img_np)
		
		# img_old = np.load(f'/home/bairong.li/240106/DRC_Deploy/output1/{fname_}'+'.npy')
		# print(np.mean(np.abs(img_old-img_np)))
		
		# img_np = (np.clip(img_np, 0, 1) * 255.0).round()
		# img_np = img_np.astype(np.uint8)
		# img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
		# cv2.imwrite(f'/home/bairong.li/240106/DRC_Deploy/output2/{fname_}'+'.png', img_np)
		
		# img_old = cv2.imread(f'/home/bairong.li/240106/DRC_Deploy/output1/{fname_}'+'.png')
		# print(np.mean(np.abs(img_old-img_np)))
	



	############################################### TRACE ##########################################

	# img = torch.rand((1, 3, 1440, 2560))
	# model = net.cpu().eval()
	# trace_model = torch.jit.trace(model, (img,))
	# torch.jit.save(trace_model, "drc_v3.pt")

	# dummy_input = (img,)
	# input_names = ("img",)
	# torch.onnx.export(net, dummy_input, 'drc_v3.onnx', input_names=input_names)

