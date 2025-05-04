from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision3d.layers import ConvBlock, build_act_layer
from image_backbone  import BasicBlock

class talk_net(nn.Module):
    def __init__(self):
        super().__init__()

        self.mask_2d = nn.Sequential(
            ConvBlock(128, 1, 3, 1, 1, 
                conv_cfg="Conv2d", norm_cfg="BatchNorm", act_cfg="Sigmoid"))
        self.mask_3d = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        
        self.comm_2d = nn.Sequential(nn.Linear(128, 128))
        self.comm_3d = nn.Sequential(nn.Linear(128, 128))

        self.summ_2d = nn.Sequential(
            ConvBlock(256, 128, 3, 1, 1, 
                conv_cfg="Conv2d", norm_cfg="BatchNorm", act_cfg="LeakyReLU"))
        self.summ_3d = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU())

        # self.over_2d = nn.Sequential(
        #     ConvBlock(128, 1, 3, 1, 1, 
        #         conv_cfg="Conv2d", norm_cfg="BatchNorm", act_cfg="Identity"))
        # self.over_3d = nn.Sequential(nn.Linear(128, 1))

    def forward(self, data_dict, img_feats_f, pcd_feats_f):
        '''
            img_feats_f:  torch.Size([1, 128, 480, 640])
            pcd_feats_f:  torch.Size([22432, 128])
        '''
        # speak layer
        mask_2d = self.mask_2d(img_feats_f)
        img_talk_fea = torch.mean(img_feats_f*mask_2d, dim=(2,3))
        mask_3d = self.mask_3d(pcd_feats_f)
        pts_talk_fea = torch.mean(pcd_feats_f*mask_3d, dim=0)

        # communication layer
        img_talk_fea = img_talk_fea + self.comm_2d(pts_talk_fea - img_talk_fea)
        pts_talk_fea = pts_talk_fea + self.comm_3d(img_talk_fea - pts_talk_fea)
        
        img_talk_fea_tile = img_talk_fea.tile((1,1,1,1))
        img_talk_fea_tile = img_talk_fea_tile.transpose(1,3)
        img_talk_fea_tile = \
            img_talk_fea_tile.tile((1,1,img_feats_f.size(2),img_feats_f.size(3)))
        pts_talk_fea_tile = pts_talk_fea.tile((pcd_feats_f.size(0),1))

        # summary layer
        img_feats_fa = torch.concatenate((img_feats_f, img_talk_fea_tile), dim=1)
        pcd_feats_fa = torch.concatenate((pcd_feats_f, pts_talk_fea_tile), dim=1)
        img_feats_fb = self.summ_2d(img_feats_fa)
        pcd_feats_fb = self.summ_3d(pcd_feats_fa)

        # weighed feature fusion
        alpha = 0.1
        img_feats_f = img_feats_fb*alpha + img_feats_f*(1-alpha)
        pcd_feats_f = pcd_feats_fb*alpha + pcd_feats_f*(1-alpha)

        # overlap prediction
        # mask_2d_pd = self.over_2d(img_feats_f)
        # mask_3d_pd = self.over_3d(pcd_feats_f)
        
        return img_feats_f, pcd_feats_f
        # return img_feats_f, pcd_feats_f, mask_2d_pd, mask_3d_pd
