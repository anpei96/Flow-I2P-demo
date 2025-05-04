import os.path as osp
import random
from typing import Optional

import cv2
import open3d as o3d
import numpy as np
from torch.utils.data import Dataset

from vision3d.array_ops import (
    apply_transform,
    compose_transforms,
    get_2d3d_correspondences_mutual,
    get_2d3d_correspondences_radius,
    get_transform_from_rotation_translation,
    inverse_transform,
    random_sample_small_transform,
)
from vision3d.utils.io import load_pickle, read_depth_image, read_image


def _get_frame_name(filename):
    _, seq_name, frame_name = filename.split(".")[0].split("/")
    seq_id = seq_name.split("-")[-1]
    frame_id = frame_name.split("_")[-1]
    output_name = f"{seq_id}-{frame_id}"
    return output_name


class I2PHardPairDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        subset: str,
        max_points: Optional[int] = None,
        return_corr_indices: bool = False,
        matching_method: str = "mutual_nearest",
        matching_radius_2d: float = 8.0,
        matching_radius_3d: float = 0.0375,
        scene_name: Optional[str] = None,
        overlap_threshold: Optional[float] = None,
        use_augmentation: bool = False,
        augmentation_noise: float = 0.005,
        scale_augmentation: bool = False,
        return_overlap_indices: bool = False,
    ):
        super().__init__()

        assert subset in ["trainval", "train", "val", "test"]
        assert matching_method in ["mutual_nearest", "radius"], f"Bad matching method: {matching_method}"

        self.dataset_dir = dataset_dir
        self.data_dir = osp.join(self.dataset_dir, "data")
        self.metadata_dir = osp.join(self.dataset_dir, "metadata")
        self.subset = subset
        self.metadata_list = load_pickle(osp.join(self.metadata_dir, f"{self.subset}-full.pkl"))

        if scene_name is not None:
            self.metadata_list = [x for x in self.metadata_list if x["scene_name"] == scene_name]

        # self.metadata_list = [x for x in self.metadata_list if "seq-11/color_019.png" in x["image_file"]]

        if overlap_threshold is not None:
            self.metadata_list = [x for x in self.metadata_list if x["overlap"] >= overlap_threshold]

        self.max_points = max_points
        self.return_corr_indices = return_corr_indices
        self.matching_method = matching_method
        self.matching_radius_2d = matching_radius_2d
        self.matching_radius_3d = matching_radius_3d
        self.overlap_threshold = overlap_threshold
        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.scale_augmentation = scale_augmentation
        self.return_overlap_indices = return_overlap_indices

        # using 5% training dataset label
        # using 20% test dataset label for the fast debug
        is_use_train_ratio = True
        # is_use_train_ratio = False
        metadata_list_new = []
        if is_use_train_ratio:
            num_all = len(self.metadata_list)
            for i in range(num_all):
                if i%20 == 0:
                    metadata_list_new.append(self.metadata_list[i])
            self.metadata_list = metadata_list_new
        
        # is_use_scene_train = True
        # is_use_scene_train = False
        # if is_use_scene_train:
        #     self.metadata_list = [x for x in self.metadata_list if x["scene_name"] == "chess"]

        # pre-compute
        metadata: dict = self.metadata_list[0]
        intrinsics_file = osp.join(self.data_dir, metadata["scene_name"], "camera-intrinsics.txt")
        intrinsics = np.loadtxt(intrinsics_file)
        fx = intrinsics[0,0]
        fy = intrinsics[1,1]
        cx = intrinsics[0,2]
        cy = intrinsics[1,2]
        image = read_image(osp.join(self.data_dir, metadata["image_file"]), as_gray=True)
        self.bear_image = np.zeros((image.shape[0], image.shape[1], 2))
        for u in range(image.shape[1]):
            for v in range(image.shape[0]):
                self.bear_image[v,u,0] = (u-cx)/fx
                self.bear_image[v,u,1] = (v-cy)/fy

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, index: int):
        data_dict = {}

        # print("index: ", index)
        metadata: dict = self.metadata_list[index]
        data_dict["scene_name"] = metadata["scene_name"]
        data_dict["image_file"] = metadata["image_file"]
        data_dict["depth_file"] = metadata["depth_file"]
        data_dict["cloud_file"] = metadata["cloud_file"]
        data_dict["overlap"] = metadata["overlap"]
        data_dict["image_id"] = _get_frame_name(metadata["image_file"])
        data_dict["cloud_id"] = _get_frame_name(metadata["cloud_file"])

        intrinsics_file = osp.join(self.data_dir, metadata["scene_name"], "camera-intrinsics.txt")
        intrinsics = np.loadtxt(intrinsics_file)
        transform = metadata["cloud_to_image"]

        # read image
        depth = read_depth_image(osp.join(self.data_dir, metadata["depth_file"])).astype(np.float)
        image = read_image(osp.join(self.data_dir, metadata["image_file"]), as_gray=True)

        data_dict["image_h"] = image.shape[0]
        data_dict["image_w"] = image.shape[1]

        # read points
        points = np.load(osp.join(self.data_dir, metadata["cloud_file"]))
        sel_indices = np.random.permutation(points.shape[0])[: self.max_points]
        if self.max_points is not None and points.shape[0] > self.max_points:
            points = points[sel_indices]

        '''
            note-0218: anpei add
        '''
        is_use_normal = True
        if is_use_normal:
            '''
                point cloud normal
            '''
            _path = str(osp.join(self.data_dir, metadata["cloud_file"]))
            _path = _path[:-4] + "_n.bin"
            normals = np.fromfile(_path, dtype=np.float32)
            normals = normals.reshape((-1,3))
            _path = str(osp.join(self.data_dir, metadata["cloud_file"]))
            _path = _path[:-4] + "_c.bin"
            curs = np.fromfile(_path, dtype=np.float32)
            curs = curs.reshape((-1,1))
            if self.max_points is not None and points.shape[0] > self.max_points:
                normals = normals[sel_indices]
                curs    = curs[sel_indices]
            # points = np.concatenate((points, normals), axis=1)

            '''
                image normal
            '''
            # _path = str(osp.join(self.data_dir, metadata["depth_file"]))
            # _path = _path[:-4] + "_n.jpg"
            _path = str(osp.join(self.data_dir, metadata["image_file"]))
            _path = _path[:-4] + "_dsine.png"

            normal_img = read_image(_path, as_gray=False)
            rgb_img = read_image(osp.join(self.data_dir, metadata["image_file"]), as_gray=False)
            rgb_img -= rgb_img.mean()
            
            # image = rgb_img*1.0
            image = np.concatenate((rgb_img, normal_img), axis=2)
            # image = normal_img

            '''
                image normal gradient
            '''
            an_rgb = cv2.imread(osp.join(self.data_dir, metadata["image_file"]))
            an_sno = cv2.imread(_path)

            sobel_x = cv2.Sobel(an_rgb, cv2.CV_16S, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(an_rgb, cv2.CV_16S, 0, 1, ksize=3)
            abs_x   = cv2.convertScaleAbs(sobel_x)
            abs_y   = cv2.convertScaleAbs(sobel_y)
            an_rgb_g = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

            sobel_x = cv2.Sobel(an_sno, cv2.CV_16S, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(an_sno, cv2.CV_16S, 0, 1, ksize=3)
            abs_x   = cv2.convertScaleAbs(sobel_x)
            abs_y   = cv2.convertScaleAbs(sobel_y)
            an_sno_g = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

            image = np.concatenate((
                image, an_rgb_g/255.0, an_sno_g/255.0), axis=2)

            # cv2.imshow("an_rgb", an_rgb)
            # cv2.imshow("an_sno", an_sno)
            # cv2.imshow("an_rgb_g", an_rgb_g)
            # cv2.imshow("an_sno_g", an_sno_g)
            # cv2.waitKey(0)

            '''
                point cloud normal gradient
            '''
            # pcd_vis = o3d.geometry.PointCloud()
            # pcd_vis.points = o3d.utility.Vector3dVector(points[:,:3])
            # pcd_vis.colors = o3d.utility.Vector3dVector(normals[:,:3])
            # show_pcd(pcd_vis)
            # assert 1==-1

        if self.use_augmentation:
            # augment point cloud
            aug_transform = random_sample_small_transform()
            center = points.mean(axis=0)
            subtract_center = get_transform_from_rotation_translation(None, -center)
            add_center = get_transform_from_rotation_translation(None, center)
            aug_transform = compose_transforms(subtract_center, aug_transform, add_center)
            if is_use_normal:
                points, normals = apply_transform(points, aug_transform, normals)
            else:
                points = apply_transform(points, aug_transform)
            inv_aug_transform = inverse_transform(aug_transform)
            transform = compose_transforms(inv_aug_transform, transform)
            points += (np.random.rand(points.shape[0], 3) - 0.5) * self.aug_noise

        self.bear_image_cur = self.bear_image * 1.0
        if self.scale_augmentation and random.random() > 0.5:
            # augment image
            scale = random.uniform(1.0, 1.2)
            raw_image_h = image.shape[0]
            raw_image_w = image.shape[1]
            new_image_h = int(raw_image_h * scale)
            new_image_w = int(raw_image_w * scale)
            start_h = new_image_h // 2 - raw_image_h // 2
            end_h = start_h + raw_image_h
            start_w = new_image_w // 2 - raw_image_w // 2
            end_w = start_w + raw_image_w
            image = cv2.resize(image, (new_image_w, new_image_h), interpolation=cv2.INTER_LINEAR)
            image = image[start_h:end_h, start_w:end_w]
            depth = cv2.resize(depth, (new_image_w, new_image_h), interpolation=cv2.INTER_LINEAR)
            depth = depth[start_h:end_h, start_w:end_w]
            intrinsics[0, 0] = intrinsics[0, 0] * scale
            intrinsics[1, 1] = intrinsics[1, 1] * scale
            self.bear_image_cur /= scale

        # build correspondences
        if self.return_corr_indices:
            if self.matching_method == "mutual_nearest":
                img_corr_pixels, pcd_corr_indices = get_2d3d_correspondences_mutual(
                    depth, points, intrinsics, transform, self.matching_radius_2d, self.matching_radius_3d
                )
            else:
                img_corr_pixels, pcd_corr_indices = get_2d3d_correspondences_radius(
                    depth, points, intrinsics, transform, self.matching_radius_2d, self.matching_radius_3d
                )
            img_corr_indices = img_corr_pixels[:, 0] * image.shape[1] + img_corr_pixels[:, 1]
            data_dict["img_corr_pixels"] = img_corr_pixels
            data_dict["img_corr_indices"] = img_corr_indices
            data_dict["pcd_corr_indices"] = pcd_corr_indices
            # print("img_corr_pixels: ", img_corr_pixels.shape)

        if self.return_overlap_indices:
            img_corr_pixels, pcd_corr_indices = get_2d3d_correspondences_radius(
                depth, points, intrinsics, transform, self.matching_radius_2d, self.matching_radius_3d
            )
            img_corr_indices = img_corr_pixels[:, 0] * image.shape[1] + img_corr_pixels[:, 1]
            img_overlap_indices = np.unique(img_corr_indices)
            pcd_overlap_indices = np.unique(pcd_corr_indices)
            img_overlap_h_pixels = img_overlap_indices // image.shape[1]
            img_overlap_w_pixels = img_overlap_indices % image.shape[1]
            img_overlap_pixels = np.stack([img_overlap_h_pixels, img_overlap_w_pixels], axis=1)
            data_dict["img_overlap_pixels"] = img_overlap_pixels
            data_dict["img_overlap_indices"] = img_overlap_indices
            data_dict["pcd_overlap_indices"] = pcd_overlap_indices

        # build data dict
        data_dict["intrinsics"] = intrinsics.astype(np.float32)
        data_dict["transform"] = transform.astype(np.float32)
        data_dict["image"] = image.astype(np.float32)
        data_dict["depth"] = depth.astype(np.float32)
        data_dict["points"] = points.astype(np.float32)
        data_dict["feats"] = np.ones(shape=(points.shape[0], 1), dtype=np.float32)

        if is_use_normal == True:
            data_dict["normals"] = normals.astype(np.float32)
            data_dict["curs"]    = curs.astype(np.float32)
            data_dict["normal_img"] = normal_img.astype(np.float32)
            
            # if disable point cloud normal features
            # data_dict["normals"] = np.ones_like(normals.astype(np.float32))
            
            # adding bearing vector
            # image = np.concatenate((image, self.bear_image_cur), axis=2)
            # data_dict["image"] = image.astype(np.float32)

            '''
                generate overlap 2d region
            '''
            is_need_generate_overlap_mask = True
            if is_need_generate_overlap_mask:
                mask_2d = np.zeros((normal_img.shape[0], normal_img.shape[1], 1), dtype=np.uint8)
                point_size = 5
                point_color = 255
                thickness = -1
                for i in range(img_corr_pixels.shape[0]):
                    u = img_corr_pixels[i,0]
                    v = img_corr_pixels[i,1]
                    cv2.circle(mask_2d, (int(v), int(u)), point_size, point_color, thickness)
                mask_2d = mask_2d.astype(np.float32)
                mask_2d = mask_2d * (1.0/255.0)
                data_dict["mask_2d_gt"] = mask_2d.astype(np.float32)

            '''
                generate overlap 3d region test --- ok
            '''
            if is_need_generate_overlap_mask:
                pcd_over_mask = np.zeros_like(points[:,0])
                colors = np.ones_like(normals)
                for i in range(img_corr_pixels.shape[0]):
                    pcd_over_mask[pcd_corr_indices[i]] = 1
                    colors[pcd_corr_indices[i], 1:] = 0
                data_dict["mask_3d_gt"] = pcd_over_mask.astype(np.float32)
                # pcd_vis = o3d.geometry.PointCloud()
                # pcd_vis.points = o3d.utility.Vector3dVector(points[:,:3])
                # pcd_vis.colors = o3d.utility.Vector3dVector(colors[:,:3])
                # show_pcd(pcd_vis)
                # print("total overlap: ", np.sum(pcd_over_mask), " / ", pcd_over_mask.shape)
                # assert 1 == -1
            return data_dict

            '''
                generate normal gradient region
                    from img_corr_pixels, pcd_corr_indices
            '''
            pcd_grad_indices = np.zeros_like(points[:,0])
            pcd_over_indices = np.zeros_like(points[:,0])
            colors = np.ones_like(normals)
            for i in range(img_corr_pixels.shape[0]):
                u = img_corr_pixels[i,0]
                v = img_corr_pixels[i,1]
                dn = np.linalg.norm(an_sno_g[u,v,:])
                pcd_over_indices[pcd_corr_indices[i]] = 1
                # colors[pcd_corr_indices[i],0] = 0
                if dn >= 10:
                    pcd_grad_indices[pcd_corr_indices[i]] = 1
                    colors[pcd_corr_indices[i],1] = 0

            pcd_vis = o3d.geometry.PointCloud()
            pcd_vis.points = o3d.utility.Vector3dVector(points[:,:3])
            pcd_vis.colors = o3d.utility.Vector3dVector(colors[:,:3])
            show_pcd(pcd_vis)

            print(np.sum(pcd_grad_indices))
            print(np.sum(pcd_over_indices))
            print(points.shape)
            print("img_corr_pixels: ", img_corr_pixels.shape)
            print("pcd_corr_indices: ", pcd_corr_indices.shape)
            assert 1==-1

            return data_dict

def show_pcd(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window("point cloud")
    render_options: o3d.visualization.RenderOption = vis.get_render_option()
    render_options.background_color = np.array([0,0,0])
    render_options.point_size = 3.0
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.run() 
