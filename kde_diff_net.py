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
        # return pdf, kernel_values
        return pdf

    def jointPdf(self, kernel_values1, kernel_values2):
        '''
            kernel_values_img:  torch.Size([1, 307200, 256])
            kernel_values_pcd:  torch.Size([1,  19268, 256])
        '''
        joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2) 
        normalization = torch.sum(joint_kernel_values, dim=(1,2)).view(-1, 1, 1) + self.epsilon
        pdf = joint_kernel_values / normalization
        return pdf

class KdeDiFF(nn.Module):
    def __init__(self, zip_num = 2, bin_num = 32):
        super().__init__()
        self.zip_num = zip_num
        self.bin_num = bin_num
        self.zip_fea_2d = nn.Sequential(
            ConvBlock(128, self.zip_num, 3, 1, 1, 
                conv_cfg="Conv2d", norm_cfg="BatchNorm", act_cfg="ReLU"))
        self.zip_fea_3d = nn.Sequential(nn.Linear(128, self.zip_num), nn.ReLU())
        self.up_fea_2d  = nn.Sequential(nn.Linear(self.zip_num, 128), nn.ReLU())
        self.up_fea_3d  = nn.Sequential(nn.Linear(self.zip_num, 128), nn.ReLU())
        self.mi = MutualInformation(num_bins=self.bin_num)

    def forward(self, data_dict, img_feats_f, pcd_feats_f):
        '''
            please note that the batch size is only one

            img_feats_f:  torch.Size([1, 128, 480, 640])
            pcd_feats_f:  torch.Size([22432, 128])

            [C,H,W] => C dimension H*W samples
            [C,N]   => C dimension N   samples
        '''
        
        '''
            step one: construct kde functions
        '''
        num_channel = self.zip_num                             # C
        img_feats_f_zip = self.zip_fea_2d(img_feats_f) 
        pcd_feats_f_zip = self.zip_fea_3d(pcd_feats_f) 
        tmp_img_f = img_feats_f_zip.reshape((num_channel,-1)) # [C,H*W]
        tmp_pcd_f = pcd_feats_f_zip.t()                       # [C,N]
        
        input_1 = tmp_img_f.t().unsqueeze(0) # [1, H*W, C]
        input_2 = tmp_pcd_f.t().unsqueeze(0) # [1,   N, C]  

        time_st = time.time()
        kde_img = torch.zeros((self.zip_num, self.bin_num)).cuda() # [C, Bin]
        kde_pcd = torch.zeros((self.zip_num, self.bin_num)).cuda() # [C, Bin]
        for c in range(self.zip_num):
            input_1_c = input_1[:,:,c:c+1]
            input_2_c = input_2[:,:,c:c+1]
            input_1_c = (input_1_c - torch.min(input_1_c))/(torch.max(input_1_c)-torch.min(input_1_c))
            input_2_c = (input_2_c - torch.min(input_2_c))/(torch.max(input_2_c)-torch.min(input_2_c))
            kde_img[c:c+1,:] = self.mi.marginalPdf(input_1_c) # [1, Bin=256]
            kde_pcd[c:c+1,:] = self.mi.marginalPdf(input_2_c) # [1, Bin=256]
        
        '''
            step two: construct self and cross matrix
                to overcome nan in the dimension-wise interaction
        '''
        Axx = torch.mm(kde_img, kde_img.t()) # [C,C] 
        Ayy = torch.mm(kde_pcd, kde_pcd.t())
        Axy = torch.mm(kde_img, kde_pcd.t())
        Axx = torch.nn.functional.normalize(Axx)
        Ayy = torch.nn.functional.normalize(Ayy)
        Axy = torch.nn.functional.normalize(Axy)

        '''
            step three: kde based feature fusion
        '''
        tmp_img_f = torch.mm(Axx+Axy, tmp_img_f) # [C,H*W]
        tmp_pcd_f = torch.mm(Ayy+Axy, tmp_pcd_f) # [C,N]
        tmp_img_f = self.up_fea_2d(tmp_img_f.t()).t()
        tmp_pcd_f = self.up_fea_3d(tmp_pcd_f.t())

        h, w = img_feats_f.size(2), img_feats_f.size(3)
        tmp_img_f = tmp_img_f.reshape((-1, h, w)).unsqueeze(0)
        
        tau = 0.1
        img_feats_f = tmp_img_f*tau + img_feats_f*(1-tau)
        pcd_feats_f = tmp_pcd_f*tau + pcd_feats_f*(1-tau)

        # time_ed = time.time()
        # print("input_1: ", input_1.size())
        # print("input_2: ", input_2.size())
        # print("cost time: ", time_ed-time_st)
        return img_feats_f, pcd_feats_f

        
