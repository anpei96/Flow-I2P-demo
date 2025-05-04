import os.path as osp
import time

import numpy as np
from config import make_cfg
from dataset import test_data_loader
from loss import EvalFunction
from anpei_model import create_model_an

from vision3d.engine import SingleTester
from vision3d.utils.io import ensure_dir
from vision3d.utils.misc import get_log_string
from vision3d.utils.parser import add_tester_args
from vision3d.utils.tensor import tensor_to_array


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)

        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg)
        loading_time = time.time() - start_time
        self.log(f"Data loader created: {loading_time:.3f}s collapsed.", level="DEBUG")
        self.log(f"Calibrate neighbors: {neighbor_limits}.")
        self.register_loader(data_loader)
    
        '''
        note-0229:
            our 2d-3d registration model *-*
        '''
        model = create_model_an(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.eval_func = EvalFunction(cfg).cuda()

        # preparation
        self.output_dir = cfg.exp.cache_dir

        # anpei save index number
        self.save_idx = 0

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.eval_func(data_dict, output_dict)
        result_dict["duration"] = output_dict["duration"]
        return result_dict

    def get_log_string(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict["scene_name"]
        image_id = data_dict["image_id"]
        cloud_id = data_dict["cloud_id"]
        message = f"{scene_name}, img: {image_id}, pcd: {cloud_id}"
        message += ", " + get_log_string(result_dict=result_dict)
        message += ", nCorr: {}".format(output_dict["corr_scores"].shape[0])
        return message

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict["scene_name"]
        image_id = data_dict["image_id"]
        cloud_id = data_dict["cloud_id"]

        ensure_dir(osp.join(self.output_dir, scene_name))
        file_name = osp.join(self.output_dir, scene_name, f"{image_id}_{cloud_id}.npz")
        np.savez_compressed(
            file_name,
            image_file=data_dict["image_file"],
            depth_file=data_dict["depth_file"],
            cloud_file=data_dict["cloud_file"],
            pcd_points=tensor_to_array(output_dict["pcd_points"]),
            pcd_points_f=tensor_to_array(output_dict["pcd_points_f"]),
            pcd_points_c=tensor_to_array(output_dict["pcd_points_c"]),
            img_num_nodes=output_dict["img_num_nodes"],
            pcd_num_nodes=output_dict["pcd_num_nodes"],
            img_node_corr_indices=tensor_to_array(output_dict["img_node_corr_indices"]),
            pcd_node_corr_indices=tensor_to_array(output_dict["pcd_node_corr_indices"]),
            img_node_corr_levels=tensor_to_array(output_dict["img_node_corr_levels"]),
            img_corr_points=tensor_to_array(output_dict["img_corr_points"]),
            pcd_corr_points=tensor_to_array(output_dict["pcd_corr_points"]),
            img_corr_pixels=tensor_to_array(output_dict["img_corr_pixels"]),
            pcd_corr_pixels=tensor_to_array(output_dict["pcd_corr_pixels"]),
            corr_scores=tensor_to_array(output_dict["corr_scores"]),
            gt_img_node_corr_indices=tensor_to_array(output_dict["gt_img_node_corr_indices"]),
            gt_pcd_node_corr_indices=tensor_to_array(output_dict["gt_pcd_node_corr_indices"]),
            gt_img_node_corr_overlaps=tensor_to_array(output_dict["gt_img_node_corr_overlaps"]),
            gt_pcd_node_corr_overlaps=tensor_to_array(output_dict["gt_pcd_node_corr_overlaps"]),
            gt_node_corr_min_overlaps=tensor_to_array(output_dict["gt_node_corr_min_overlaps"]),
            gt_node_corr_max_overlaps=tensor_to_array(output_dict["gt_node_corr_max_overlaps"]),
            transform=tensor_to_array(data_dict["transform"]),
            intrinsics=tensor_to_array(data_dict["intrinsics"]),
            overlap=data_dict["overlap"],
        )

        # anpei visulization of 2d-3d registration
        import cv2 as cv
        import torch
        from vision3d.ops import apply_transform

        base_path = "/media/anpei/DiskC/2d-3d-reg-work/codes/2D3DMATR-main/datasets/7Scenes/data/"
        save_path = "/media/anpei/DiskC/2d-3d-reg-work/codes/" + \
            "2D3DMATR-main/scene_plus/results/seven/"
        import os
        if os.path.exists(save_path) == False:
            os.mkdir(save_path)

        is_need_visulization = True
        # is_need_visulization = False
        if is_need_visulization:
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
            
            # vis_rgb_pts_img = np.concatenate((rgb_image, pts_image), axis=1)
            vis_rgb_pts_img = np.concatenate((rgb_image, pts_image), axis=0)
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
                cv.circle(vis_rgb_pts_img, (int(u), int(v)+img_h), 1, (0,255,255), -1)

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
                        (u_img, v_img), (u_pts, v_pts+img_h), (0,0,255), 1)
                if d <= th:
                    cv.line(vis_rgb_pts_img, 
                        (u_img, v_img), (u_pts, v_pts+img_h), (0,255,0), 1)
            
            '''
                4. visulize PIR/IR in the image
            ''' 
            # res_string = "IR: "+ str(result_dict['IR'].cpu().numpy())
            res_string = "IR: "+ format(result_dict['IR'].cpu().numpy(), '.3f')
            cv.putText(vis_rgb_pts_img, res_string, 
                (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            # print("img_corr_pixels: ", img_corr_pixels.shape)
            # print("pcd_corr_pixels: ", pcd_corr_pixels.shape)
            # print("result_dict")
            # print(result_dict)

            # cv.imshow("vis_rgb_pts_img", vis_rgb_pts_img)
            # cv.imshow("vis_mix_img", vis_mix_img)
            # cv.waitKey(0)

            '''
                5. save the visulization image
            ''' 
            save_name = save_path + str(self.save_idx) + ".png"
            self.save_idx += 1
            # cv.imwrite(save_name, vis_rgb_pts_img)
            # print("save image in the path: ", save_name)
            # assert 1==-1

        '''
            t-SNE visulization
        '''
        from sklearn.preprocessing import StandardScaler 
        import matplotlib.pyplot as plt 
        from sklearn.manifold import TSNE 
        import torch.nn.functional as F

        is_need_visulization_manifold = True
        if is_need_visulization_manifold:
            # img_feats_f = output_dict["img_feats_c"].cpu().numpy()
            # pcd_feats_f = output_dict["pcd_feats_c"].cpu().numpy()

            img_feats_f = output_dict["img_feats_f"].cpu().numpy()
            pcd_feats_f = output_dict["pcd_feats_f"].cpu().numpy()

            # img_feats_f = output_dict["img_feats_f_0"]
            # pcd_feats_f = output_dict["pcd_feats_f_0"]
            # img_feats_f = F.normalize(img_feats_f, p=2, dim=1)
            # pcd_feats_f = F.normalize(pcd_feats_f, p=2, dim=1)
            # img_feats_f = img_feats_f[:,:2].cpu().numpy()
            # pcd_feats_f = pcd_feats_f[:,:2].cpu().numpy()

            # down-sample img_feats_f
            img_feats_fx = []
            num = img_feats_f.shape[0]
            for i in range(num):
                if i % 10 == 0:
                    img_feats_fx.append(img_feats_f[i])
            img_feats_fx = np.array(img_feats_fx)

            source = img_feats_fx
            target = pcd_feats_f
            print("source: ", source.shape)
            print("target: ", target.shape)
            scalar = StandardScaler()

            source_len = source.shape[0]   
            all_data = np.concatenate((source,target),axis=0)
            all_data = scalar.fit_transform(all_data)
            # DR_data = all_data[:,:2]
            nn = 0
            DR_data = all_data[:,nn:nn+2]

            # tsne = TSNE(n_components=2,perplexity=30,n_iter=1000)
            # DR_data = tsne.fit_transform(all_data)
            DR_source = DR_data[0:source_len,:]
            DR_target = DR_data[source_len:,:]

            #source的为红色的⚪
            plt.scatter(DR_source[:,0],DR_source[:,1],color="#FF0000",marker="o",label="image feature")  
            #target的为蓝色的▲
            plt.scatter(DR_target[:,0],DR_target[:,1],color="#0000FF",marker="^",label="point feature")
            #去掉坐标轴的刻度线
            plt.xticks([])  # 去x坐标刻度
            plt.yticks([])  # 去y坐标刻度
            #plt.axis('off')  # 去坐标轴
            #显示出图例 
            plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
            plt.show()

            print("img_feats_f: ", img_feats_f.shape)
            print("pcd_feats_f: ", pcd_feats_f.shape)
            assert 1==-1

def main():
    add_tester_args()
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == "__main__":
    main()
