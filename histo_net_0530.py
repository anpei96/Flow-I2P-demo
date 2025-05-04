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
        self.bins_2 = nn.Parameter(torch.linspace(0, 1, num_bins).float(), requires_grad=False)
        self.bins_1 = nn.Parameter(torch.linspace(0, 4, num_bins).float(), requires_grad=False)

    def marginalPdf(self, values):
        residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5*(residuals / self.sigma).pow(2))
        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization
        return pdf, kernel_values

    def marginalPdf_v2(self, values, v_min, v_max):
        bins_this = nn.Parameter(torch.linspace(v_min, v_max, self.num_bins).float(), requires_grad=False)
        bins_this = bins_this.cuda()
        residuals = values - bins_this.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5*(residuals / self.sigma).pow(2))
        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization
        return pdf, kernel_values
    
    def marginalPdf_v3(self, values, is_input_img):
        if is_input_img == True:
            bins_this = self.bins_1
        else:
            bins_this = self.bins_2
        residuals = values - bins_this.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5*(residuals / self.sigma).pow(2))
        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization
        return pdf, kernel_values

class histo_net(nn.Module):
    def __init__(self):
        super().__init__()

        self.grayscale_2d = nn.Sequential(
            ConvBlock(128, 1, 3, 1, 1, 
                conv_cfg="Conv2d", norm_cfg="BatchNorm", act_cfg="ReLU"))
        self.grayscale_3d = nn.Sequential(nn.Linear(128, 1), nn.ReLU())
        
        self.mi = MutualInformation()

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

        # compute distribution
        ta = time.time()
        v_min = 0
        v_max = 4 #int(torch.max(input_1).detach().cpu().numpy())
        pdf_img, kernel_values_img = \
            self.mi.marginalPdf_v3(input_1, is_input_img=True)
        # pdf_img = torch.histc(input_1, bins=256, min=v_min, max=v_max)

        v_min = 0
        v_max = 1 #int(torch.max(input_2).detach().cpu().numpy())
        pdf_pcd, kernel_values_pcd = \
            self.mi.marginalPdf_v3(input_2, is_input_img=False)
        # pdf_pcd = torch.histc(input_2, bins=256, min=v_min, max=v_max)
        tb = time.time()

        print("mutal info time: ", tb-ta)
        print("pdf_img: ", pdf_img.size())
        print("pdf_pcd: ", pdf_pcd.size())
        print("img_max: ", torch.max(input_1))
        print("pcd_max: ", torch.max(input_2))
        # print("kernel_values_img: ", kernel_values_img.size())
        # print("kernel_values_pcd: ", kernel_values_pcd.size())
        # assert 1 == -1

        return img_feats_f, pcd_feats_f

        
