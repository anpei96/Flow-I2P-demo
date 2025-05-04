import os.path as osp
import random
from typing import Optional

import cv2
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


class FangWeiHardPairDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        subset: str,
        max_points: Optional[int] = None,
        return_corr_indices: bool = False,
        matching_method: str = "mutual_nearest",
        matching_radius_2d: float = 8.0,
        matching_radius_3d: float = 0.1,
        scene_name: Optional[str] = None,
        overlap_threshold: Optional[float] = None,
        use_augmentation: bool = False,
        augmentation_noise: float = 0.005,
        scale_augmentation: bool = False,
    ):
        super().__init__()

        assert subset in ["train", "val", "test"], f"Bad subset name: {subset}."
        assert matching_method in ["mutual_nearest", "radius"], f"Bad matching method: {matching_method}"

        self.dataset_dir = dataset_dir
        self.data_dir = osp.join(self.dataset_dir, "data")
        self.metadata_dir = osp.join(self.dataset_dir, "metadata")
        self.subset = subset
        self.metadata_list = load_pickle(osp.join(self.metadata_dir, f"{self.subset}.pkl"))


        if scene_name is not None:
            self.metadata_list = [x for x in self.metadata_list if x["scene_name"] == scene_name]

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

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, index: int):
        data_dict = {}

        metadata: dict = self.metadata_list[index]
        data_dict["scene_name"] = metadata["scene_name"]
        data_dict["image_file"] = metadata["image_file"]
        data_dict["depth_file"] = metadata["depth_file"]
        data_dict["cloud_file"] = metadata["cloud_file"]
        data_dict["overlap"] = metadata["overlap"]
        data_dict["image_id"] = osp.basename(metadata["image_file"]).split(".")[0].split("_")[1]
        data_dict["cloud_id"] = osp.basename(metadata["cloud_file"]).split(".")[0].split("_")[1]

        intrinsics_file = osp.join(self.data_dir, metadata["scene_name"], "camera-intrinsics.txt")
        intrinsics = np.loadtxt(intrinsics_file)

        transform_file = osp.join(self.data_dir, metadata["cloud_to_image"])
        transform = np.loadtxt(transform_file)

        # read image
        depth = read_depth_image(osp.join(self.data_dir, metadata["depth_file"])).astype(np.float32)
        image = read_image(osp.join(self.data_dir, metadata["image_file"]), as_gray=True)
        rgb_image = read_image(osp.join(self.data_dir, metadata["image_file"]), as_gray=False)

        data_dict["image_h"] = image.shape[0]
        data_dict["image_w"] = image.shape[1]

        # read points
        rgb_points_path = data_dict["scene_name"] + "/rgb_cloud_" + "points_" + data_dict["cloud_id"] + ".npy"
        # points = np.load(osp.join(self.data_dir, metadata["cloud_file"]))
        rgb_points = np.load(osp.join(self.data_dir, rgb_points_path)) # (N, 1, 3)
        points = rgb_points.reshape(rgb_points.shape[0], 3) # (N, 3)
        rgb_pcd_colors_path = data_dict["scene_name"] + "/rgb_cloud_" +  "colors_" + data_dict["cloud_id"] + ".npy"
        rgb_pcd_colors = np.load(osp.join(self.data_dir, rgb_pcd_colors_path)) # (N, 1, 3)
        points_color = rgb_pcd_colors.reshape(rgb_pcd_colors.shape[0], 3) # (N, 3)

        data_dict["cloud_file"] = self.data_dir + "/" + rgb_points_path

        if self.max_points is not None and points.shape[0] > self.max_points:
            sel_indices = np.random.permutation(points.shape[0])[: self.max_points]
            points = points[sel_indices]
            points_color = points_color[sel_indices]

        '''
            note-0218: anpei add
        '''
        is_use_normal = True
        if is_use_normal:
            '''
                point cloud normal
            '''
            _path = self.data_dir + "/" + rgb_points_path
            _path = _path[:-4] + "_n.bin"
            normals = np.fromfile(_path, dtype=np.float32)
            normals = normals.reshape((-1,3))
            curs = np.zeros_like(normals[:,0])
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
            # print("==========>")

        if self.use_augmentation:
            # augment point cloud
            aug_transform = random_sample_small_transform()
            center = points.mean(axis=0)
            subtract_center = get_transform_from_rotation_translation(None, -center)
            add_center = get_transform_from_rotation_translation(None, center)
            aug_transform = compose_transforms(subtract_center, aug_transform, add_center)
            points = apply_transform(points, aug_transform)
            inv_aug_transform = inverse_transform(aug_transform)
            transform = compose_transforms(inv_aug_transform, transform)
            points += (np.random.rand(points.shape[0], 3) - 0.5) * self.aug_noise

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
            rgb_image = cv2.resize(image, (new_image_w, new_image_h), interpolation=cv2.INTER_LINEAR)
            rgb_image = rgb_image[start_h:end_h, start_w:end_w]
            depth = cv2.resize(depth, (new_image_w, new_image_h), interpolation=cv2.INTER_LINEAR)
            depth = depth[start_h:end_h, start_w:end_w]
            intrinsics[0, 0] = intrinsics[0, 0] * scale
            intrinsics[1, 1] = intrinsics[1, 1] * scale

        # image -= image.mean()
        # rgb_image -= rgb_image.mean()

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
        
        # build data dict
        data_dict["intrinsics"] = intrinsics.astype(np.float32)
        data_dict["transform"] = transform.astype(np.float32)
        data_dict["image"] = image.astype(np.float32)
        data_dict["rgb_image"] = rgb_image.astype(np.float32)
        data_dict["depth"] = depth.astype(np.float32)
        data_dict["points"] = points.astype(np.float32)
        data_dict["points_colors"] = points_color.astype(np.float32)
        data_dict["feats"] = np.ones(shape=(points.shape[0], 1), dtype=np.float32)

        # return data_dict
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
