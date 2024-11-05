import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import json
from pathlib import Path
import os
import utils.pc_util as pc_util

from scipy.spatial import cKDTree

from models import loss_helper_unlabeled

from models.singleton_utils import SingletonType

# pseudo_instances_db = {}
labeled_isntances_db = {}


class AugmentHelper(metaclass=SingletonType): 
    def __init__(self, dataset_config, mixup_ratio=0.5):
        self.dc = dataset_config
        self.mixup_ratio = mixup_ratio
        
    def get_gt_instance_by_sem_cls(self, sem_cls: int, equal=True):
        ret_instances = []
        
        for instance in labeled_isntances_db.values():
            if equal:
                if instance['sem_cls'] == sem_cls:
                    ret_instances.append(instance)
            else:
                if instance['sem_cls'] != sem_cls:
                    ret_instances.append(instance)
                
        return ret_instances

    def add_to_labeled_instance_db_scannet(self, idx, point_cloud, semantic_labels, instance_bboxes):
        for instance_bbox in instance_bboxes:
            modified_bbox = instance_bbox.copy()
            modified_bbox = np.insert(modified_bbox, 6, 0.0)
            
            _, filtered_pc = filter_sem_points_in_bbox(point_cloud, modified_bbox, semantic_labels)
            if filtered_pc.shape[0] < 100:
                continue
            key = (idx, *modified_bbox[:3])
            
            labeled_isntances_db[key] = {
                "point_cloud": filtered_pc,
                "center": modified_bbox[:3],
                "size": modified_bbox[3:6],
                "heading": 0.0,
                "sem_cls": modified_bbox[-1],
            }
            
    def add_to_labeled_instance_db_sunrgbd(self, idx, point_cloud, bboxes):
        """

        Args:
            idx (_type_): _description_
            point_cloud (_type_): _description_
            bboxes (_type_): x,y,z,l,w,h,r,cls
        """
        for instance_bbox in bboxes:
            modified_bbox = instance_bbox.copy()
            modified_bbox[3:6] *= 2 

            _, filtered_pc = filter_points_in_bbox(point_cloud, modified_bbox[:-1])
            if filtered_pc.shape[0] < 100:
                continue
            key = (idx, *modified_bbox[:3])

            labeled_isntances_db[key] = {
                "point_cloud": filtered_pc,
                "center": modified_bbox[:3],
                "size": modified_bbox[3:6],
                "heading": modified_bbox[6],
                "sem_cls": modified_bbox[-1],
            }

    ####################################### mix up #######################################
    def mix_up_1(self, batch_data_label, teacher_end_points, config_dict):
        """
        random select one box from pseudo instance
        exchange by ground truth instance from labeled instance database with same sem class
        resize to target box size
        """
        end_points = self.get_pseudo_instances(batch_data_label, teacher_end_points, config_dict)
        chosen_pseudo_instance, chosen_pseudo_instance_mask = self.get_random_pseudo_instance_batch(
                                                                    end_points, config_dict)

        gt_instance = self.get_gt_instance_by_pseudo_instance_batch(chosen_pseudo_instance, equal=True)
        
        # render_point_cloud_matplotlib(gt_instance[0]['point_cloud'], save_prefix="gt_instance")
        
        resized_gt_instance = self.resize_instance_batch_gpu(gt_instance, chosen_pseudo_instance)
        
        # render_point_cloud_matplotlib(resized_gt_instance[0]['point_cloud'], save_prefix="resized_gt_instance")
        
        mixup_instances = self.mixup_2_instances(
            chosen_pseudo_instance,
            resized_gt_instance,
            ratio=1.0
        )
        mixup_point_clouds = self.replace_pc_by_instance(
            batch_data_label['point_clouds'], 
            mixup_instances,
            chosen_pseudo_instance_mask
        )

        batch_data_label['point_clouds'] = mixup_point_clouds
        return batch_data_label

    def mix_up_2(self, batch_data_label, teacher_end_points, config_dict):
        """
        random select one box from pseudo instance
        select ground truth instance from labeled instance database with same sem class
        resize to target box size
        mixup two instance and resample with A% and 100-A%
        """
        end_points = self.get_pseudo_instances(batch_data_label, teacher_end_points, config_dict)
        chosen_pseudo_instance, chosen_pseudo_instance_mask = self.get_random_pseudo_instance_batch(
                                                                    end_points, config_dict)

        gt_instance = self.get_gt_instance_by_pseudo_instance_batch(chosen_pseudo_instance, equal=True)
        
        # if gt_instance[0] is not None:
        #     render_point_cloud_matplotlib(gt_instance[0]['point_cloud'], save_prefix="gt_instance")
        
        resized_gt_instance = self.resize_instance_batch_gpu(gt_instance, chosen_pseudo_instance)
        
        # if resized_gt_instance[0] is not None:
        #     render_point_cloud_matplotlib(resized_gt_instance[0]['point_cloud'], save_prefix="resized_gt_instance")

        mixup_ratio = np.random.uniform(0.3, 0.7)

        mixup_instances = self.mixup_2_instances(
            chosen_pseudo_instance,
            resized_gt_instance,
            ratio=mixup_ratio
        )
        mixup_point_clouds = self.replace_pc_by_instance(
            batch_data_label['point_clouds'], 
            mixup_instances,
            chosen_pseudo_instance_mask
        )

        batch_data_label['point_clouds'] = mixup_point_clouds
        return batch_data_label

    def mix_up_3(self, batch_data_label, teacher_end_points, config_dict):
        """
        random select one box from pseudo instance
        select ground truth instance from labeled isntance database with different sem class
        resize to target box size
        mixup two instance and resample with A% and 100-A% A > 50
        """
        end_points = self.get_pseudo_instances(batch_data_label, teacher_end_points, config_dict)
        chosen_pseudo_instance, chosen_pseudo_instance_mask = self.get_random_pseudo_instance_batch(
                                                                    end_points, config_dict)

        gt_instance = self.get_gt_instance_by_pseudo_instance_batch(chosen_pseudo_instance, equal=False)
        
        resized_gt_instance = self.resize_instance_batch_gpu(gt_instance, chosen_pseudo_instance)

        mixup_ratio = np.random.uniform(0.2, 0.5)

        mixup_instances = self.mixup_2_instances(
            chosen_pseudo_instance,
            resized_gt_instance,
            ratio=mixup_ratio
        )
        mixup_point_clouds = self.replace_pc_by_instance(
            batch_data_label['point_clouds'], 
            mixup_instances,
            chosen_pseudo_instance_mask
        )

        batch_data_label['point_clouds'] = mixup_point_clouds
        return batch_data_label
    

    def get_pseudo_instances(self, batch_data_label, teacher_end_points, config_dict):
        ema_end_points = teacher_end_points.copy()

        for key in batch_data_label:
            ema_end_points[key] = batch_data_label[key]

        labeled_num = torch.nonzero(ema_end_points['supervised_mask']).squeeze(1).shape[0]
        pred_center = ema_end_points['center'][labeled_num:]
        pred_sem_cls = ema_end_points['sem_cls_scores'][labeled_num:]
        pred_objectness = ema_end_points['objectness_scores'][labeled_num:]
        pred_heading_scores = ema_end_points['heading_scores'][labeled_num:]
        pred_heading_residuals = ema_end_points['heading_residuals'][labeled_num:]
        pred_size_scores = ema_end_points['size_scores'][labeled_num:]
        pred_size_residuals = ema_end_points['size_residuals'][labeled_num:]
        pred_vote_xyz = ema_end_points['aggregated_vote_xyz'][labeled_num:]

        loss_helper_unlabeled.get_pseudo_labels(
            ema_end_points,
            ema_end_points,
            pred_center,
            pred_sem_cls,
            pred_objectness,
            pred_heading_scores,
            pred_heading_residuals,
            pred_size_scores,
            pred_size_residuals,
            pred_vote_xyz,
            config_dict
        )

        return ema_end_points
    
    def get_random_pseudo_instance_batch(self, end_points, config_dict):
        labeled_num = torch.nonzero(end_points['supervised_mask']).squeeze(1).shape[0]
        pseudo_mask = end_points['pseudo_mask']

        selected_pseudo_instances = []
        selected_pseudo_instance_masks = []

        instance_centers = self.trans_center(
            end_points['center'][labeled_num:],
            end_points['flip_x_axis'][labeled_num:],
            end_points['flip_y_axis'][labeled_num:],
            end_points['rot_mat'][labeled_num:], 
            end_points['scale'][labeled_num:]
        )
        instance_sizes = self.trans_size(
            end_points['size'][labeled_num:],
            end_points['scale'][labeled_num:]
        )
        instance_headings = self.trans_heading(
            end_points['heading'][labeled_num:],
            end_points['flip_x_axis'][labeled_num:],
            end_points['flip_y_axis'][labeled_num:],
            end_points['rot_angle'][labeled_num:]
        )

        for i in range(pseudo_mask.shape[0]):
            ones_indices = np.nonzero(pseudo_mask[i])[0]
            if len(ones_indices) > 0:
                
                # selected_index = np.random.choice(ones_indices)
                
                ones_instances = []
                ones_instance_masks = []
                
                # gt_densities = compute_density()
                # gt_num_points = compute_num_points()
                
                for selected_index in ones_indices:

                    instance_center = instance_centers[i, selected_index]
                    instance_size = instance_sizes[i, selected_index]
                    instance_heading = instance_headings[i, selected_index]
                    instance_semcls_scores = end_points['sem_cls_scores'][labeled_num + i, selected_index]
                    instance_semcls = torch.argmax(instance_semcls_scores)

                    if config_dict['dataset'] == 'scannet':
                        instance_semcls = config_dict['dataset_config'].nyu40ids[instance_semcls]
                    elif config_dict['dataset'] == 'sunrgbd':
                        pass  # TODO
                    else:
                        pass

                    instance_mask, instance_point_cloud = filter_points_in_bbox_gpu(
                        point_cloud=end_points['point_clouds'][labeled_num + i],
                        bbox=torch.cat([instance_center, instance_size, instance_heading.unsqueeze(0)], dim=0)
                    )

                    # filter density > mean_density - std_density * 2
                    # density = instance_point_cloud.shape[0] / (instance_size[0] * instance_size[1] * instance_size[2])
                    # density = density.detach().cpu().item()
                    # num_points = instance_point_cloud.shape[0]
                    
                    # mean_density = np.mean(gt_densities.get(instance_semcls, [0, 0]))
                    # std_density = np.std(gt_densities.get(instance_semcls, [0, 0]))
                    
                    # mean_num_points = np.mean(gt_num_points.get(instance_semcls, [0, 0]))
                    # std_num_points = np.std(gt_num_points.get(instance_semcls, [0, 0]))
                    
                    # if mean_num_points - std_num_points < num_points < mean_num_points + std_num_points and \
                    #     mean_density - std_density < density < mean_density + std_density and \
                    #     num_points > 50:
                    if True:
                        ones_instances.append(dict(
                            point_cloud=instance_point_cloud,
                            center=instance_center,
                            size=instance_size,
                            heading=instance_heading,
                            sem_cls=instance_semcls
                        ))

                        ones_instance_masks.append(instance_mask)
                
                
                if len(ones_instances) > 0:
                    instances = []
                    instance_masks = []
                    
                    # # random single
                    # selected_idx = np.random.randint(len(ones_instances))
                    # instances.append(ones_instances[selected_idx])
                    # instance_masks.append(ones_instance_masks[selected_idx])
                    
                    # partition multiple
                    # ratio = 0.5
                    ratio = self.mixup_ratio
                    num_selected = int(len(ones_instances) * ratio) + 1
                    selected_indices = np.random.choice(len(ones_instances), num_selected, replace=False)

                    for idx in selected_indices:
                        instances.append(ones_instances[idx])
                        instance_masks.append(ones_instance_masks[idx])
                    
                    # default
                    selected_pseudo_instances.append(instances)
                    selected_pseudo_instance_masks.append(instance_masks)
                    
                    
                else:
                    selected_pseudo_instances.append(None)
                    selected_pseudo_instance_masks.append(None)

        return selected_pseudo_instances, selected_pseudo_instance_masks

    def get_gt_instance_by_pseudo_instance_batch(self, target_instances, equal=True):
        ret_instances = []
        
        for scene_instances in target_instances:  # batch
            if scene_instances is None:
                ret_instances.append(None)
                continue
            
            scene_gt_instances = []
            
            for target_instance in scene_instances:
                # if target_instance is None:
                #     ret_instances.append(None)
                #     continue
                target_semcls = target_instance['sem_cls']

                gt_instances = self.get_gt_instance_by_sem_cls(target_semcls, equal=equal)
                
                if len(gt_instances) > 0:
                    scene_gt_instances.append(np.random.choice(gt_instances))
                else:
                    scene_gt_instances.append(None)
            
            ret_instances.append(scene_gt_instances)

        return ret_instances

    def resize_instance_batch_gpu(self, source_instances, target_instances):
        ret_ret_instances = []
        for i in range(len(target_instances)):
            if source_instances[i] is None or target_instances[i] is None:
                ret_ret_instances.append(None)
                continue
            ret_instances = []
            for j in range(len(target_instances[i])):
                if source_instances[i][j] is None or target_instances[i][j] is None:
                    ret_instances.append(None)
                else:
                    target_instance = target_instances[i][j]
                    source_instance = source_instances[i][j].copy()
                    target_size = target_instance['size'].detach().cpu().numpy()
                    source_size = source_instance['size']
                    scale_factors = target_size / source_size

                    shifted_point_cloud = source_instance['point_cloud'].copy()
                    shifted_point_cloud[:, 0:3] -= source_instance['center']
                    shifted_point_cloud[:, -1] -= source_instance['center'][-1]

                    resized_shifted_point_cloud = shifted_point_cloud.copy()
                    resized_shifted_point_cloud[:, 0:3] *= scale_factors
                    resized_shifted_point_cloud[:, -1] *= scale_factors[-1]

                    resized_point_cloud = resized_shifted_point_cloud.copy()
                    resized_point_cloud[:, 0:3] += source_instance['center']
                    resized_point_cloud[:, -1] += source_instance['center'][-1]

                    ret_instances.append(dict(
                        point_cloud=resized_point_cloud,
                        center=source_instance['center'],
                        size=target_size,
                        heading=source_instance['heading'],
                        sem_cls=source_instance['sem_cls']
                    ))
            ret_ret_instances.append(ret_instances)
        
        return ret_ret_instances


    def mixup_2_instances(self, pseudo_instances, gt_instances, ratio=0.5):
        mixup_mixup_instances = []
        
        for i in range(len(pseudo_instances)):
            if gt_instances[i] is None:
                mixup_mixup_instances.append(pseudo_instances[i])
                continue
            mixup_instances = []
            for j in range(len(pseudo_instances[i])):
                if gt_instances[i][j] is None:
                    mixup_instances.append(pseudo_instances[i][j])
                    continue
                mixup_instance = pseudo_instances[i][j].copy()
                pseudo_instance_pc = pseudo_instances[i][j]['point_cloud']
                gt_instance_pc = gt_instances[i][j]['point_cloud']

                # shift to original point
                gt_instance_pc[:, 0:3] -= gt_instances[i][j]['center']
                gt_instance_pc[:, -1] -= gt_instances[i][j]['center'][-1]
                
                # rotate gt -> pseudo
                rot_angle = pseudo_instances[i][j]['heading'].detach().cpu().item() - gt_instances[i][j]['heading']
                rot_mat = pc_util.rotz(rot_angle)
                gt_instance_pc[:, 0:3] = np.dot(gt_instance_pc[:, 0:3], np.transpose(rot_mat))
                
                # shift to target center
                gt_instance_pc[:, 0:3] += pseudo_instances[i][j]['center'].detach().cpu().numpy()
                gt_instance_pc[:, -1] += pseudo_instances[i][j]['center'].detach().cpu().numpy()[-1]
                
                # if i == 0:
                #     render_point_cloud_matplotlib(gt_instance_pc, save_prefix="rot_gt_instance")
                #     render_point_cloud_matplotlib(pseudo_instance_pc.detach().cpu().numpy(), save_prefix="pseudo_instance")

                N = pseudo_instance_pc.shape[0]
        
                num_from_gt = int(ratio * gt_instance_pc.shape[0])
                num_from_pseudo = max(int((1 - ratio) * pseudo_instance_pc.shape[0]), N - num_from_gt)  # num_gt + num_pseudo >= N

                sampled_tensor_cloud = pseudo_instance_pc[torch.randperm(N)[:num_from_pseudo]]

                sampled_ndarray_cloud = gt_instance_pc[np.random.choice(gt_instance_pc.shape[0], num_from_gt, replace=False)]
                sampled_ndarray_cloud = torch.from_numpy(sampled_ndarray_cloud).to(pseudo_instance_pc.device)

                mixed_cloud = torch.cat([sampled_tensor_cloud, sampled_ndarray_cloud], dim=0)
                assert mixed_cloud.shape[0] >= N
                mixup_instance['point_cloud'] = mixed_cloud
                mixup_instances.append(mixup_instance)
                
                # if i == 0:
                #     render_point_cloud_matplotlib(mixed_cloud.detach().cpu().numpy(), save_prefix="mixup_instance")
            mixup_mixup_instances.append(mixup_instances)                
        
        return mixup_mixup_instances
    
    def replace_pc_by_instance(self, source_point_cloud, instances, choices):
        unlabeled_num = len(instances)
        labeled_num = source_point_cloud.shape[0] - unlabeled_num
        replaced_point_cloud = source_point_cloud.clone()

        for i in range(unlabeled_num):
            if instances[i] is None:
                continue
            # instance_pc = instances[i]['point_cloud']
            
            # assert instance_pc.shape[0] == torch.sum(choices[i]), \
            #     f"Instance point cloud size {instance_pc.shape[0]} doesn't match the number of selected points {torch.sum(choices[i])}."

            # replaced_point_cloud[i + labeled_num][choices[i]] = instance_pc
            
            
            point_cloud_to_replace = replaced_point_cloud[i + labeled_num].clone()
            
            # 删除 replaced_point_cloud 中 choices 指定的点
            mask = torch.ones(replaced_point_cloud[i + labeled_num].shape[0], dtype=torch.bool)
            for choice in choices[i]:
                mask[choice] = False
            point_cloud_to_replace = point_cloud_to_replace[mask]

            # 将 instance_pc 的所有点追加到 replaced_point_cloud 中
            for instance in instances[i]:
                point_cloud_to_replace = torch.cat((point_cloud_to_replace, instance['point_cloud']), dim=0)

            # resample to keep same size
            indices = torch.randperm(point_cloud_to_replace.shape[0])[:source_point_cloud.shape[1]]
            point_cloud_to_replace = point_cloud_to_replace[indices]
            
            replaced_point_cloud[i + labeled_num] = point_cloud_to_replace

        return replaced_point_cloud
    
    def trans_center(self, center, flip_x_axis, flip_y_axis, rot_mat, scale_ratio):
        """
        teacher -> student
        """
        center_clone = center.clone()
        inds_to_flip_x_axis = torch.nonzero(flip_x_axis).squeeze(1)
        center_clone[inds_to_flip_x_axis, :, 0] = -center[inds_to_flip_x_axis, :, 0]

        inds_to_flip_y_axis = torch.nonzero(flip_y_axis).squeeze(1)
        center_clone[inds_to_flip_y_axis, :, 1] = -center[inds_to_flip_y_axis, :, 1]

        center_clone = torch.bmm(center_clone, rot_mat.transpose(1, 2))  # (B, num_proposal, 3)
        center_clone = center_clone * scale_ratio  # (B, K, 3) * (B, 1, 3)
        return center_clone

    def trans_size(self, size, scale_ratio):
        """
        teacher -> student
        """
        return size * scale_ratio

    def trans_heading(self, heading, flip_x_axis, flip_y_axis, rot_angle):
        """
        teacher -> student
        """
        angle = heading.clone()
        inds_to_flip_x_axis = torch.nonzero(flip_x_axis).squeeze(1)
        angle[inds_to_flip_x_axis, :] = np.pi - angle[inds_to_flip_x_axis, :]
        inds_to_flip_y_axis = torch.nonzero(flip_y_axis).squeeze(1)
        angle[inds_to_flip_y_axis, :] = -angle[inds_to_flip_y_axis, :]
        angle = angle - rot_angle.unsqueeze(-1)
        return angle


def filter_sem_points_in_bbox(point_cloud, bbox, sem_labels):
    """
    筛选出位于旋转包围盒（OBB）内的点云。
    
    参数:
    - point_cloud: (N, 3) 点云数组，每一行表示一个点的 (x, y, z) 坐标。
    - bbox: 包围盒参数 [x_center, y_center, z_center, length, width, height, yaw]
    
    返回:
    - filtered_points: 在包围盒内的点云。
    """
    # 解析 bbox 参数
    x_c, y_c, z_c, l, w, h, yaw, semcls = bbox

    # 计算包围盒的旋转矩阵（绕 Z 轴旋转）
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1]
    ])

    # 将点云转换到包围盒的局部坐标系
    # 平移中心点
    shifted_points = point_cloud[:, 0:3] - np.array([x_c, y_c, z_c])
    # 应用旋转矩阵
    local_points = shifted_points @ rotation_matrix.T

    # 过滤出在包围盒尺寸范围内的点
    mask = (
        (np.abs(local_points[:, 0]) <= l / 2) &
        (np.abs(local_points[:, 1]) <= w / 2) &
        (np.abs(local_points[:, 2]) <= h / 2) &
        (sem_labels == semcls)
    )
    filtered_points = point_cloud[mask]

    return mask, filtered_points

def filter_points_in_bbox(point_cloud, bbox):
    """
    筛选出位于旋转包围盒（OBB）内的点云。
    
    参数:
    - point_cloud: (N, 3) 点云数组，每一行表示一个点的 (x, y, z) 坐标。
    - bbox: 包围盒参数 [x_center, y_center, z_center, length, width, height, yaw]
    
    返回:
    - filtered_points: 在包围盒内的点云。
    """
    # 解析 bbox 参数
    x_c, y_c, z_c, l, w, h, yaw = bbox

    # 计算包围盒的旋转矩阵（绕 Z 轴旋转）
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1]
    ])

    # 将点云转换到包围盒的局部坐标系
    # 平移中心点
    shifted_points = point_cloud[:, 0:3] - np.array([x_c, y_c, z_c])
    # 应用旋转矩阵
    local_points = shifted_points @ rotation_matrix.T

    # 过滤出在包围盒尺寸范围内的点
    mask = (
        (np.abs(local_points[:, 0]) <= l / 2) &
        (np.abs(local_points[:, 1]) <= w / 2) &
        (np.abs(local_points[:, 2]) <= h / 2)
    )
    filtered_points = point_cloud[mask]
    return mask, filtered_points

def filter_points_in_bbox_gpu(point_cloud, bbox):
    """
    筛选出位于旋转包围盒（OBB）内的点云，支持张量计算并可在 GPU 上加速。

    参数:
    - point_cloud: (N, 3) 的 tensor，每一行表示一个点的 (x, y, z) 坐标。
    - bbox: 包围盒参数 tensor [x_center, y_center, z_center, length, width, height, yaw]。

    返回:
    - mask: (N,) 的布尔型 tensor，指示每个点是否在包围盒内。
    - filtered_points: 在包围盒内的点云 tensor。
    """
    # 解析 bbox 参数
    x_c, y_c, z_c, l, w, h, yaw = bbox

    # 计算旋转矩阵（绕 Z 轴旋转）
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    rotation_matrix = torch.tensor([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1]
    ], device=point_cloud.device)  # 确保与点云的设备一致

    # 将点云转换到包围盒的局部坐标系
    shifted_points = point_cloud[:, 0:3] - torch.tensor([x_c, y_c, z_c], device=point_cloud.device)
    local_points = shifted_points @ rotation_matrix.T  # 应用旋转矩阵

    # 构造掩码，筛选出位于包围盒内的点
    mask = (
        (torch.abs(local_points[:, 0]) <= l / 2) &
        (torch.abs(local_points[:, 1]) <= w / 2) &
        (torch.abs(local_points[:, 2]) <= h / 2)
    )

    # 根据掩码过滤点云
    filtered_points = point_cloud[mask]

    return mask, filtered_points