from typing import Union

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from vision3d.layers import ConvBlock, build_act_layer
from image_backbone  import BasicBlock

'''
    https://gitcode.com/connorlee77/pytorch-mutual-information/blob/master/MutualInformation.py
'''
class MutualInformation(nn.Module):
    def __init__(self, sigma=0.1, num_bins=256, normalize=True):
        super(MutualInformation, self).__init__()
        self.sigma = sigma
        self.num_bins = num_bins
        self.normalize = normalize
        self.epsilon = 1e-10
        self.bins = nn.Parameter(torch.linspace(0, 1, num_bins).float(), requires_grad=False)

    def marginalPdf(self, values):
        residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5*(residuals / self.sigma).pow(2))
        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization
        return pdf, kernel_values

    def jointPdf(self, kernel_values1, kernel_values2):
        '''
            kernel_values_img:  torch.Size([1, 307200, 256])
            kernel_values_pcd:  torch.Size([1,  19268, 256])
        '''
        joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2) 
        normalization = torch.sum(joint_kernel_values, dim=(1,2)).view(-1, 1, 1) + self.epsilon
        pdf = joint_kernel_values / normalization
        return pdf

class histo_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.grayscale_2d = nn.Sequential(
            ConvBlock(128, 1, 3, 1, 1, 
                conv_cfg="Conv2d", norm_cfg="BatchNorm", act_cfg="ReLU"))
        self.grayscale_3d = nn.Sequential(nn.Linear(128, 1), nn.ReLU())
        self.mi = MutualInformation()
        self.ch_att_2d = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
        self.ch_att_3d = nn.Sequential(nn.Linear(512, 128), nn.ReLU())

    def forward(self, data_dict, img_feats_f, pcd_feats_f):
        '''
            please note that the batch size is only one

            img_feats_f:  torch.Size([1, 128, 480, 640])
            pcd_feats_f:  torch.Size([22432, 128])

            [C,H,W] => C dimension H*W samples
            [C,N]   => C dimension N   samples
        '''
        # histogram layer
        # C dimension => 1 dimension
        num_channel = 1
        img_feats_f_gray = self.grayscale_2d(img_feats_f) 
        pcd_feats_f_gray = self.grayscale_3d(pcd_feats_f) 
        tmp_img_f = img_feats_f_gray.reshape((num_channel,-1)) # [C,H*W]
        tmp_pcd_f = pcd_feats_f_gray.t()                       # [C,N]
        input_1 = tmp_img_f.t().unsqueeze(0) # [1, H*W, C]
        input_2 = tmp_pcd_f.t().unsqueeze(0) # [1,   N, C]

        # compute distribution with input normalization
        input_1 = (input_1 - torch.min(input_1))/(torch.max(input_1)-torch.min(input_1))
        input_2 = (input_2 - torch.min(input_2))/(torch.max(input_2)-torch.min(input_2))
        pdf_img, _ = self.mi.marginalPdf(input_1) # [1, Bin=256]
        pdf_pcd, _ = self.mi.marginalPdf(input_2) # [1, Bin=256]

        # compute diffusion interaction feature --- channel attetion
        diff_pdf  = torch.concatenate((pdf_img, pdf_pcd), dim=1) # [1, 512]
        ch_att_2d = self.ch_att_2d(diff_pdf) # [1, C]
        ch_att_3d = self.ch_att_3d(diff_pdf) # [1, C]

        ch_att_2d_tile = ch_att_2d.tile((1,1,1,1))
        ch_att_2d_tile = ch_att_2d_tile.transpose(1,3)
        ch_att_2d_tile = \
            ch_att_2d_tile.tile((1,1,img_feats_f.size(2),img_feats_f.size(3)))
        ch_att_3d_tile = ch_att_3d.tile((pcd_feats_f.size(0),1))

        img_feats_fd = img_feats_f*ch_att_2d_tile
        pcd_feats_fd = pcd_feats_f*ch_att_3d_tile

        # weighed feature fusion
        alpha = 0.1
        img_feats_f = img_feats_fd*alpha + img_feats_f*(1-alpha)
        pcd_feats_f = pcd_feats_fd*alpha + pcd_feats_f*(1-alpha)

        # print("ch_att_2d_tile: ", ch_att_2d_tile.size())
        # print("ch_att_3d_tile: ", ch_att_3d_tile.size())
        # print("img_feats_f: ",    img_feats_f.size())
        # print("pcd_feats_f: ",    pcd_feats_f.size())
        # assert 1 == -1
        return img_feats_f, pcd_feats_f

        
