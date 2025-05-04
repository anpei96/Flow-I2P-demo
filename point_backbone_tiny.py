import torch
import torch.nn as nn

from vision3d.layers import KPConvBlock, KPResidualBlock, UnaryBlockPackMode
from vision3d.ops import knn_interpolate_pack_mode

class PointBackbone_tiny(nn.Module):
    def __init__(self, input_dim, output_dim, init_dim, kernel_size, init_radius, init_sigma):
        super().__init__()
        
        self.encoder_1 = KPConvBlock(input_dim, init_dim, kernel_size, init_radius, init_sigma)
        self.encoder_2 = KPResidualBlock(init_dim, init_dim * 2, kernel_size, init_radius, init_sigma)
        self.encoder_3 = KPResidualBlock(init_dim * 2, init_dim * 4, kernel_size, init_radius, init_sigma)
        self.encoder_4 = KPResidualBlock(init_dim * 4, init_dim * 4, kernel_size, init_radius, init_sigma)
        self.out_proj = nn.Linear(init_dim * 4, output_dim)

    def forward(self, feats, data_dict):
        feats_list = []

        points_list = data_dict["points"]
        neighbors_list = data_dict["neighbors"]
        subsampling_list = data_dict["subsampling"]
        upsampling_list = data_dict["upsampling"]

        feats_s1 = feats
        feats_s1 = self.encoder_1(points_list[0], points_list[0], feats_s1, neighbors_list[0])
        feats_s1 = self.encoder_2(points_list[0], points_list[0], feats_s1, neighbors_list[0])
        feats_s1 = self.encoder_3(points_list[0], points_list[0], feats_s1, neighbors_list[0])
        feats_s1 = self.encoder_4(points_list[0], points_list[0], feats_s1, neighbors_list[0])
        feats_s1 = self.out_proj(feats_s1)

        feats_list.append(feats_s1)
        feats_list.reverse()
        return feats_list