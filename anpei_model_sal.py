import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision3d.engine.checkpoint import load_state_dict
from anpei_model import create_model_an
from anpei_saliency_net import create_model_saliency_net
from vision3d.models.geotransformer import SuperPointMatchingMutualTopk, SuperPointProposalGenerator
from vision3d.ops import (
    back_project,
    batch_mutual_topk_select,
    create_meshgrid,
    index_select,
    pairwise_cosine_similarity,
    point_to_node_partition,
    render,
)

class Diff2D3D_SAL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.matching_radius_2d = cfg.model.ground_truth_matching_radius_2d
        self.matching_radius_3d = cfg.model.ground_truth_matching_radius_3d
        self.pcd_num_points_in_patch = cfg.model.pcd_num_points_in_patch

        # fixed for now
        self.img_h_c = 24
        self.img_w_c = 32
        self.img_num_levels_c = 3
        self.overlap_threshold = 0.3
        self.pcd_min_node_size = 5

        self.coarse_matching = SuperPointMatchingMutualTopk(
            cfg.model.coarse_matching.num_correspondences,
            k=cfg.model.coarse_matching.topk,
            threshold=cfg.model.coarse_matching.similarity_threshold,
        )

        # load the baseline matcher
        model = create_model_an(cfg)
        self.baseline_model = model.cuda().eval()
        self.baseline_model.eval()
        base_path = "/media/anpei/DiskC/2d-3d-reg-work/codes/2D3DMATR-main/scene_plus/model_zoos/"
        state_dict = torch.load(base_path+"new_base_5p5p.pth", map_location=torch.device("cpu"))
        load_state_dict(self.baseline_model, state_dict["model"], strict=True)
        print("==> load it ...")

        # initialize the saliency matcher
        model_sal = create_model_saliency_net()
        self.saliency_model = model_sal.cuda()

    def forward(self, data_dict):
        output_dict = self.baseline_model(data_dict)
        output_dict = self.post_process_generate_corres(
            output_dict["img_feats_c"].detach(),
            output_dict["pcd_feats_c"].detach(),
            output_dict["img_feats_f"].detach(),
            output_dict["pcd_feats_f"].detach(),
            output_dict)
        output_dict = self.saliency_model(data_dict, output_dict)
        return output_dict

    def post_process_generate_corres(self, img_feats_c, pcd_feats_c,
        img_feats_f, pcd_feats_f, output_dict):
        (
            img_node_corr_indices,
            pcd_node_corr_indices,
            node_corr_scores,
        ) = self.coarse_matching(img_feats_c, pcd_feats_c, output_dict["img_node_masks"], output_dict["pcd_node_masks"])
        img_node_corr_levels = output_dict["img_node_levels"][img_node_corr_indices]

        output_dict["img_node_corr_indices"] = img_node_corr_indices
        output_dict["pcd_node_corr_indices"] = pcd_node_corr_indices
        output_dict["img_node_corr_levels"] = img_node_corr_levels

        pcd_padded_feats_f = torch.cat([pcd_feats_f, torch.zeros_like(pcd_feats_f[:1])], dim=0)

        # 7. Extract patch correspondences
        all_img_corr_indices = []
        all_pcd_corr_indices = []

        for i in range(self.img_num_levels_c):
            node_corr_masks = torch.eq(img_node_corr_levels, i)

            if node_corr_masks.sum().item() == 0:
                continue

            cur_img_node_corr_indices = img_node_corr_indices[node_corr_masks] - output_dict["all_img_total_nodes"][i]
            cur_pcd_node_corr_indices = pcd_node_corr_indices[node_corr_masks]

            img_node_knn_points  = output_dict["all_img_node_knn_points"][i]
            img_node_knn_pixels  = output_dict["all_img_node_knn_pixels"][i]
            img_node_knn_indices = output_dict["all_img_node_knn_indices"][i]

            img_node_corr_knn_indices = index_select(img_node_knn_indices, cur_img_node_corr_indices, dim=0)
            img_node_corr_knn_masks = torch.ones_like(img_node_corr_knn_indices, dtype=torch.bool)
            img_node_corr_knn_feats = index_select(img_feats_f, img_node_corr_knn_indices, dim=0)

            pcd_node_corr_knn_indices = output_dict["pcd_node_knn_indices"][cur_pcd_node_corr_indices]  # (P, Kc)
            pcd_node_corr_knn_masks = output_dict["pcd_node_knn_masks"][cur_pcd_node_corr_indices]  # (P, Kc)
            pcd_node_corr_knn_feats = index_select(pcd_padded_feats_f, pcd_node_corr_knn_indices, dim=0)

            similarity_mat = pairwise_cosine_similarity(
                img_node_corr_knn_feats, pcd_node_corr_knn_feats, normalized=True
            )

            batch_indices, row_indices, col_indices, _ = batch_mutual_topk_select(
                similarity_mat,
                k=1,
                row_masks=img_node_corr_knn_masks,
                col_masks=pcd_node_corr_knn_masks,
                threshold=0.75,
                largest=True,
                mutual=True,
            )

            img_corr_indices = img_node_corr_knn_indices[batch_indices, row_indices]
            pcd_corr_indices = pcd_node_corr_knn_indices[batch_indices, col_indices]

            all_img_corr_indices.append(img_corr_indices)
            all_pcd_corr_indices.append(pcd_corr_indices)

        img_corr_indices = torch.cat(all_img_corr_indices, dim=0)
        pcd_corr_indices = torch.cat(all_pcd_corr_indices, dim=0)

        # duplicate removal
        num_points_f = output_dict["pcd_points_f"].shape[0]
        corr_indices = img_corr_indices * num_points_f + pcd_corr_indices
        unique_corr_indices = torch.unique(corr_indices)
        img_corr_indices = torch.div(unique_corr_indices, num_points_f, rounding_mode="floor")
        pcd_corr_indices = unique_corr_indices % num_points_f

        img_points_f = output_dict["img_points_f"].view(-1, 3)
        img_pixels_f = output_dict["img_pixels_f"].view(-1, 2)
        img_corr_points = img_points_f[img_corr_indices]
        img_corr_pixels = img_pixels_f[img_corr_indices]
        pcd_corr_points = output_dict["pcd_points_f"][pcd_corr_indices]
        pcd_corr_pixels = output_dict["pcd_points_f"][pcd_corr_indices]
        img_corr_feats = img_feats_f[img_corr_indices]
        pcd_corr_feats = pcd_feats_f[pcd_corr_indices]
        corr_scores = (img_corr_feats * pcd_corr_feats).sum(1)

        output_dict["img_corr_points"] = img_corr_points
        output_dict["img_corr_pixels"] = img_corr_pixels
        output_dict["img_corr_indices"] = img_corr_indices
        output_dict["pcd_corr_points"] = pcd_corr_points
        output_dict["pcd_corr_pixels"] = pcd_corr_pixels
        output_dict["pcd_corr_indices"] = pcd_corr_indices
        output_dict["corr_scores"] = corr_scores
        return output_dict

def create_model_an_sal(cfg):
    model = Diff2D3D_SAL(cfg)
    return model

def main():
    from config import make_cfg
    cfg = make_cfg()
    model = create_model_an_sal(cfg)
    print(model.state_dict().keys())
    print(model)

if __name__ == "__main__":
    main()
