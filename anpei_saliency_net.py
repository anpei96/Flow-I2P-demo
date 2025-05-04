import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2 as cv
import numpy as np
from vision3d.ops import apply_transform
from vision3d.utils.tensor import tensor_to_array, array_to_tensor
from vision3d.layers import ConvBlock
from image_backbone_tiny import ImageBackbone_tiny

base_path = "/media/anpei/DiskC/2d-3d-reg-work/codes/2D3DMATR-main/datasets/7Scenes/data/"
save_path = "/media/anpei/DiskC/2d-3d-reg-work/codes/" + \
    "2D3DMATR-main/scene_plus/results/seven/"

def visulization_correspondence(data_dict, output_dict):
    image_file = data_dict["image_file"]
    depth_file = data_dict["depth_file"]
    cloud_file = data_dict["cloud_file"]
    transform  = data_dict["transform"]
    intrinsics = data_dict["intrinsics"]

    '''
        1. visulize rgb/depth image
    '''

    rgb_image = cv.imread(base_path + image_file)
    img_h = rgb_image.shape[0]
    img_w = rgb_image.shape[1]

    pts = np.load(base_path + cloud_file)
    pts = apply_transform(torch.tensor(pts).cuda().float(), transform)
    pix = (torch.matmul(intrinsics, pts.T)).T
    pix = pix.cpu().numpy()
    dep = pix[:,2:3]
    pix = pix/dep
    pix = pix.astype(np.int)
    num = pix.shape[0]
    d_max, d_min = np.max(dep), np.min(dep)

    pts_image = np.zeros_like(rgb_image)
    for i in range(num):
        u = int(pix[i,0])
        v = int(pix[i,1])
        d = dep[i]
        d = int((d-d_min)/(d_max-d_min)*255)
        if ((u < 0) | (u >= img_w)):
            continue
        if ((v < 0) | (v >= img_h)):
            continue
        cv.circle(pts_image, (int(u), int(v)), 3, (d,d,255-d), -1)
    
    vis_rgb_pts_img = np.concatenate((rgb_image, pts_image), axis=1)
    vis_mix_img = cv.addWeighted(rgb_image, 0.5, pts_image, 0.5, 0)

    '''
        2. visulize 2d/3d corner points
    '''
    img_corr_pixels=tensor_to_array(output_dict["img_corr_pixels"])
    pcd_corr_pixels=tensor_to_array(output_dict["pcd_corr_pixels"])
    num_pts = img_corr_pixels.shape[0]
    for i in range(num_pts):
        u = int(img_corr_pixels[i,0])
        v = int(img_corr_pixels[i,1])
        cv.circle(vis_rgb_pts_img, (int(v), int(u)), 1, (0,255,255), -1)
    
    pts_corr = apply_transform(
        torch.tensor(pcd_corr_pixels).cuda().float(), transform)
    pix_corr = (torch.matmul(intrinsics, pts_corr.T)).T
    pix_corr = pix_corr.cpu().numpy()
    dep_corr = pix_corr[:,2:3]
    pix_corr = pix_corr/dep_corr
    pix_corr = pix_corr.astype(np.int)
    for i in range(num_pts):
        u = int(pix_corr[i,0])
        v = int(pix_corr[i,1])
        cv.circle(vis_rgb_pts_img, (int(u)+img_w, int(v)), 1, (0,255,255), -1)

    '''
        3. visulize 2d/3d point correspondence
    '''     
    for i in range(num_pts):
        u_img = int(img_corr_pixels[i,1])
        v_img = int(img_corr_pixels[i,0])
        u_pts = int(pix_corr[i,0])
        v_pts = int(pix_corr[i,1])
        d = np.abs(u_img - u_pts) + np.abs(v_img - v_pts)
        th = 15
        if d > th:
            cv.line(vis_rgb_pts_img, 
                (u_img, v_img), (u_pts+img_w, v_pts), (0,0,255), 1)
        if d <= th:
            cv.line(vis_rgb_pts_img, 
                (u_img, v_img), (u_pts+img_w, v_pts), (0,255,0), 1)
    
    '''
        4. visulize PIR/IR in the image
    ''' 
    # res_string = "IR: "+ str(result_dict['IR'].cpu().numpy())
    # res_string = "IR: "+ format(result_dict['IR'].cpu().numpy(), '.3f')
    # cv.putText(vis_rgb_pts_img, res_string, 
    #     (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # print("img_corr_pixels: ", img_corr_pixels.shape)
    # print("pcd_corr_pixels: ", pcd_corr_pixels.shape)
    # print("result_dict")
    # print(result_dict)

    cv.imshow("vis_rgb_pts_img", vis_rgb_pts_img)
    cv.imshow("vis_mix_img", vis_mix_img)
    cv.waitKey(0)

def create_model_saliency_net():
    saliency_model = co_ob_saliency_net()
    return saliency_model

class co_ob_saliency_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_2d = nn.Sequential(
            ConvBlock(
            1, 64, 3, 1, 1, 
            conv_cfg="Conv2d", norm_cfg="BatchNorm", act_cfg="Identity"),
            ConvBlock(
            64, 16, 3, 1, 1, 
            conv_cfg="Conv2d", norm_cfg="BatchNorm", act_cfg="Identity"),
            ConvBlock(
            16, 1, 3, 1, 1, 
            conv_cfg="Conv2d", norm_cfg="BatchNorm", act_cfg="Identity"),
        )
        
    def get_initial_2d_mask(self, mask_2d, output_dict):
        img_corr_pixels = tensor_to_array(output_dict["img_corr_pixels"])
        corr_scores = tensor_to_array(output_dict["corr_scores"])
        num_pts = img_corr_pixels.shape[0]
        for i in range(num_pts):
            u = int(img_corr_pixels[i,0])
            v = int(img_corr_pixels[i,1])
            cv.circle(mask_2d, (int(v), int(u)), 3, int(255*corr_scores[i]), -1)
        mask_2d = cv.medianBlur(mask_2d, 3)
        return mask_2d

    def forward(self, data_dict, output_dict):
        '''
        using such things
            output_dict["img_corr_points"] = img_corr_points
            output_dict["img_corr_pixels"] = img_corr_pixels
            output_dict["img_corr_indices"] = img_corr_indices
            output_dict["pcd_corr_points"] = pcd_corr_points
            output_dict["pcd_corr_pixels"] = pcd_corr_pixels
            output_dict["pcd_corr_indices"] = pcd_corr_indices
            output_dict["corr_scores"] = corr_scores
        '''
        # step 1. get initial point/ mask 
        #   using output_dict["img_corr_pixels"] output_dict["pcd_corr_indices"]
        h_img = data_dict["image"].size(1)
        w_img = data_dict["image"].size(2)
        img_2d_fea_map = output_dict["img_2d_fea_map"]
        pts_3d_fea_map = output_dict["pts_3d_fea_map"]
        mask_2d = np.zeros((h_img, w_img), dtype=np.uint8)
        mask_2d = self.get_initial_2d_mask(mask_2d, output_dict)
        mask_2d_tensor = mask_2d.reshape((1,1,h_img, w_img))/255.0
        mask_2d_tensor = (array_to_tensor(mask_2d_tensor)).float().cuda()

        # step 2. predict the accurate co-observable region
        #   using mask_2d_tensor, data_dict["image"]
        img_2d = data_dict["image"].unsqueeze(1).detach() 
        img_2d = img_2d.transpose(1, -1)
        img_2d = img_2d.squeeze(-1)
        fea_2d = torch.concat(
            (mask_2d_tensor, img_2d, img_2d_fea_map), dim=1)
        cof_2d = self.net_2d(mask_2d_tensor)
        output_dict["cov_2d_pd"] = cof_2d

        # is_need_debug = True
        is_need_debug = False
        if is_need_debug:
            # print("img_corr_indices: ", output_dict["img_corr_indices"].size())
            # print(output_dict["img_corr_indices"])
            # print("pcd_corr_indices: ", output_dict["pcd_corr_indices"].size())
            # print(output_dict["pcd_corr_indices"])
            # print("corr_scores: ", output_dict["corr_scores"].size())
            # print(output_dict["corr_scores"])
            
            mask_2d_vis = cv.applyColorMap(mask_2d, cv.COLORMAP_JET)
            cv.imshow("mask_2d_vis", mask_2d_vis)
            # cv.waitKey(0)
            visulization_correspondence(data_dict, output_dict)
            assert 1==-1

        return output_dict