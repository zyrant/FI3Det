# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import warnings
from os import path as osp

import numpy as np

from mmdet3d.core import (
    instance_seg_eval, instance_seg_eval_v2, show_result_v2, show_seg_result)
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmseg.datasets import DATASETS as SEG_DATASETS
from mmdet.datasets  import DATASETS
from mmdet3d.datasets.custom_3d  import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose
import mmcv
import torch
from mmdet3d.core.bbox.iou_calculators import axis_aligned_bbox_overlaps_3d

@DATASETS.register_module()
class ScanNetDataset_INC(Custom3DDataset):
    r"""ScanNet Dataset for Detection Task.

    This class serves as the API for experiments on the ScanNet Dataset.

    Please refer to the `github repo <https://github.com/ScanNet/ScanNet>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Depth' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """
    CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')


    def __init__(self,
                 data_root,
                 ann_file,
                 pesudo_ann_file=None,
                 pipeline=None,
                 classes=None,
                 modality=dict(use_camera=False, use_depth=True),
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False,
                 increment_pkl=False,
                 increament_type='old',
                 anno_type='9_1_increment',
                 mode='alpha',
                 progressive_mode=False,
                 total_stages=None,
                 current_stage=0,
                 stage_size = [3, 3, 3],
                 **kwargs):

        self.increament_type = str(increament_type)
        self.anno_type = str(anno_type)
        self.num_novel_classes = int(anno_type.split('_')[0])
        self.num_base_classes = len(self.CLASSES) - self.num_novel_classes
        self.increment_pkl = increment_pkl
        self.temp_box_type = box_type_3d
        self.progressive_mode = progressive_mode
        self.total_stages = total_stages
        self.current_stage = current_stage
        self.stage_size = stage_size
        
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert 'use_camera' in self.modality and \
               'use_depth' in self.modality
        assert self.modality['use_camera'] or self.modality['use_depth']

        if mode == 'alpha':
            self.SORTED_CLASSES = sorted(self.CLASSES)
        elif mode == 'direct':
            self.SORTED_CLASSES = self.CLASSES
        elif mode == 'random':
            self.SORTED_CLASSES = np.random.permutation(self.CLASSES).tolist()
        elif mode == 'count':
            self.SORTED_CLASSES = ('chair','door','garbagebin','cabinet','table','window','picture','desk','sofa',
                                   'sink','bed','bookshelf','curtain','counter','toilet','refrigerator','showercurtrain','bathtub')

        self.num_total_classes = len(self.SORTED_CLASSES)
        self.num_novel_classes = int(anno_type.split('_')[0])
        self.num_base_classes = self.num_total_classes - self.num_novel_classes


        if self.progressive_mode:
            assert self.total_stages is not None

            custom_stage_sizes = self.stage_size
            assert sum(custom_stage_sizes) == self.num_novel_classes, \
                "Sum of custom_stage_sizes must equal num_novel_classes"

            self.current_num_classes = (
                self.num_base_classes + sum(custom_stage_sizes[:self.current_stage])
            )
            self.past_num_classes = self.num_base_classes + sum(custom_stage_sizes[:self.current_stage - 1])            
        else:
            self.current_num_classes = self.num_total_classes
            self.past_num_classes = self.num_base_classes  

        self.BASE_CLASSES = self.SORTED_CLASSES[:self.num_base_classes]
        self.CUR_CLASSES = self.SORTED_CLASSES[:self.current_num_classes]
        self.NOVEL_CLASSES = self.SORTED_CLASSES[self.num_base_classes:self.current_num_classes]
        self.CUR_NOVEL_CLASSES = self.SORTED_CLASSES[self.past_num_classes:self.current_num_classes]


        print(f"[Stage {self.current_stage}] Active classes: {self.CUR_CLASSES}")
        self.pesudo_ann_file = pesudo_ann_file
        

        if self.pesudo_ann_file is not None:
            if hasattr(self.file_client, 'get_local_path'):
                with self.file_client.get_local_path(self.pesudo_ann_file) as local_path:
                    self.pesduo_data_infos = self.load_pesudo_annotations(open(local_path, 'rb'))
            else:
                warnings.warn(
                    'The used MMCV version does not have get_local_path. '
                    f'We treat the {self.pesudo_ann_file} as local paths and it '
                    'might cause errors if the path is not a local path. '
                    'Please use MMCV>= 1.3.16 if you meet errors.')
                self.pesduo_data_infos = self.load_pesudo_annotations(self.pesudo_ann_file)
        
        self.data_infos = self.filter_annos(self.data_infos, test_mode)
    

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format='pkl')
        if self.increment_pkl:
            data = data[self.anno_type][f'{self.num_novel_classes}_' + self.increament_type]

        return data

    def filter_annos(self, data, test_mode):
        if not self.progressive_mode:
            return data

        filtered = []
        for item in data:
            annos = item['annos']
            gt_classes = annos.get('class', None)
            gt_boxes = annos.get('gt_boxes_upright_depth', None)
            gt_num = annos.get('gt_num', 0)
            gt_name = annos.get('name', None)
            locations = annos.get('location', None)
            dimensions = annos.get('dimensions', None)
            if gt_num == 0 or gt_classes is None:
                filtered.append(item)
                continue

            cls_names = [self.CLASSES[i] for i in gt_classes]

            if test_mode:
                keep_mask = [c in self.CUR_CLASSES for c in cls_names]
            else:
                keep_mask = [c in self.CUR_NOVEL_CLASSES for c in cls_names]
            keep_mask = np.array(keep_mask, dtype=bool)

            if keep_mask.sum() > 0:
                annos['class'] = gt_classes[keep_mask]
                annos['gt_num'] = int(keep_mask.sum())
                annos['gt_boxes_upright_depth'] = gt_boxes[keep_mask]
                annos['location'] = locations[keep_mask]
                annos['dimensions'] = dimensions[keep_mask]
                annos['index'] = np.arange(
                    annos['gt_num'], dtype=np.int32)
                annos['name'] = gt_name[keep_mask]
                item['annos'] = annos
                
                filtered.append(item)

        print(f"Progressive Stage {self.current_stage}: "
            f"Retained {len(filtered)} scenes, filtered annotations per scene.")
        return filtered

    def load_pesudo_annotations(self, pesduo_ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        # loading data from a file-like object needs file format
        return mmcv.load(pesduo_ann_file, file_format='pkl')

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - img_prefix (str, optional): Prefix of image files.
                - img_info (dict, optional): Image info.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = osp.join(self.data_root, info['pts_path'])
        input_dict = dict(sample_idx=sample_idx)

        if self.modality['use_depth']:
            input_dict['pts_filename'] = pts_filename
            input_dict['file_name'] = pts_filename

        if self.modality['use_camera']:
            img_info = []
            for img_path in info['img_paths']:
                img_info.append(
                    dict(filename=osp.join(self.data_root, img_path)))
            intrinsic = info['intrinsics']
            axis_align_matrix = self._get_axis_align_matrix(info)
            depth2img = []
            for extrinsic in info['extrinsics']:
                depth2img.append(
                    intrinsic @ np.linalg.inv(axis_align_matrix @ extrinsic))

            input_dict['img_prefix'] = None
            input_dict['img_info'] = img_info
            input_dict['depth2img'] = depth2img

        if not self.test_mode:
            annos = self.get_ann_info(index)
            if self.pesudo_ann_file is not None:
                annos = self.get_pesudo_ann_info(index, annos, input_dict['pts_filename'])
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
                return None
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`DepthInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - pts_instance_mask_path (str): Path of instance masks.
                - pts_semantic_mask_path (str): Path of semantic masks.
                - axis_align_matrix (np.ndarray): Transformation matrix for
                    global scene alignment.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.int64)
            gt_name = [self.CLASSES[i] for i in gt_labels_3d]    

        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.int64)
            gt_name = ['background']

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        pts_instance_mask_path = osp.join(self.data_root,
                                          info['pts_instance_mask_path'])
        pts_semantic_mask_path = osp.join(self.data_root,
                                          info['pts_semantic_mask_path'])

        axis_align_matrix = self._get_axis_align_matrix(info)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            pts_instance_mask_path=pts_instance_mask_path,
            pts_semantic_mask_path=pts_semantic_mask_path,
            axis_align_matrix=axis_align_matrix,
            gt_names = gt_name)
        return anns_results
    
    def get_pesudo_ann_info(self, index, annos, pts_filename):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`DepthInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - pts_instance_mask_path (str): Path of instance masks.
                - pts_semantic_mask_path (str): Path of semantic masks.
                - axis_align_matrix (np.ndarray): Transformation matrix for
                    global scene alignment.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.pesduo_data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.int64)
            gt_names = [self.CLASSES[i] for i in gt_labels_3d]
            
            if 'feat_proto' in info['annos']:
                gt_feats = info['annos']['feat_proto'].astype(np.float32)
            else:
                gt_feats = np.zeros((len(gt_labels_3d), 256), dtype=np.float32)

        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.int64)
            gt_names = ['background']
            gt_feats = np.zeros((0, 6), dtype=np.float32)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)


        if len(gt_labels_3d) > 0:
            act_gt_bboxes_3d = annos['gt_bboxes_3d']
            act_gt_labels_3d = annos['gt_labels_3d']
            act_gt_names = annos['gt_names']
            gt_bboxes_3d_tran = torch.cat((gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1)[:, :6]
            act_gt_bboxes_3d_tran = torch.cat((act_gt_bboxes_3d.gravity_center, act_gt_bboxes_3d.tensor[:, 3:]), dim=1)[:, :6]
            overlaps = axis_aligned_bbox_overlaps_3d(self._bbox_to_loss(gt_bboxes_3d_tran), 
                                    self._bbox_to_loss(act_gt_bboxes_3d_tran))

            max_overlaps, _ = overlaps.max(dim=1) 


            pseudo_keep_mask = (max_overlaps < 0.25).numpy().astype(bool)
            pseudo_overlap_mask = (max_overlaps >= 0.25).numpy().astype(bool)

            new_boxes = []
            new_labels = []
            new_names = []
            new_feats = []

            # <0.25 ：LABEL -1
            if pseudo_keep_mask.sum() > 0:
                boxes_keep = gt_bboxes_3d[pseudo_keep_mask]
                labels_keep = np.full((len(boxes_keep),), -1, dtype=np.int64)
                names_keep = np.array(['objects'] * len(boxes_keep))
                feats_keep = gt_feats[pseudo_keep_mask]

                new_boxes.append(boxes_keep)
                new_labels.append(labels_keep)
                new_names.append(names_keep)
                new_feats.append(feats_keep)

            # >=0.25 LABEL -2
            if pseudo_overlap_mask.sum() > 0:
                boxes_overlap = gt_bboxes_3d[pseudo_overlap_mask]
                labels_overlap = np.full((len(boxes_overlap),), -2, dtype=np.int64)
                names_overlap = np.array(['overlap'] * len(boxes_overlap))
                feats_overlap = gt_feats[pseudo_overlap_mask]

                new_boxes.append(boxes_overlap)
                new_labels.append(labels_overlap)
                new_names.append(names_overlap)
                new_feats.append(feats_overlap)

            if len(new_boxes) > 0:
                gt_bboxes_3d = act_gt_bboxes_3d.cat([act_gt_bboxes_3d] + new_boxes)
                gt_labels_3d = np.concatenate([act_gt_labels_3d] + new_labels, axis=0)
                gt_names = np.concatenate([act_gt_names] + new_names, axis=0)
                gt_feats = np.concatenate(
                    [np.zeros((len(act_gt_labels_3d), gt_feats.shape[-1]), dtype=np.float32)] + new_feats,
                    axis=0
                )
            else:
                gt_bboxes_3d = act_gt_bboxes_3d
                gt_labels_3d = act_gt_labels_3d
                gt_names = act_gt_names
                gt_feats = np.zeros((len(act_gt_labels_3d), gt_feats.shape[-1]), dtype=np.float32)


        annos['gt_bboxes_3d'] = gt_bboxes_3d
        annos['gt_labels_3d'] = gt_labels_3d
        annos['gt_names'] = gt_names
        annos['gt_feats'] = gt_feats

        return annos
    
    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned iou loss format.
        Args:
            bbox (Tensor): 3D box of shape (N, 6).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6).
        """
        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)

    def prepare_test_data(self, index):
        """Prepare data for testing.

        We should take axis_align_matrix from self.data_infos since we need
            to align point clouds.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        
        # take the axis_align_matrix from data_infos
        # input_dict['ann_info'] = dict(
        #     axis_align_matrix=self._get_axis_align_matrix(
        #         self.data_infos[index]))
        
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos
        if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
            return None
        
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @staticmethod
    def _get_axis_align_matrix(info):
        """Get axis_align_matrix from info. If not exist, return identity mat.

        Args:
            info (dict): one data info term.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        if 'axis_align_matrix' in info['annos'].keys():
            return info['annos']['axis_align_matrix'].astype(np.float32)
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
            dict(type='GlobalAlignment', rotation_axis=2),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._build_default_pipeline()
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points', load_annos=True).numpy()
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            gt_bboxes = gt_bboxes.corners.numpy() if len(gt_bboxes) else None
            gt_labels = self.get_ann_info(i)['gt_labels_3d']
            pred_bboxes = result['boxes_3d']
            pred_bboxes = pred_bboxes.corners.numpy() if len(pred_bboxes) else None
            pred_labels = result['labels_3d']
            show_result_v2(points, gt_bboxes, gt_labels,
                           pred_bboxes, pred_labels, out_dir, file_name)

    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str], optional): Metrics to be evaluated.
                Defaults to None.
            iou_thr (list[float]): AP IoU thresholds. Defaults to (0.25, 0.5).
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        from mmdet3d.core.evaluation import indoor_eval
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'
        gt_annos = [info['annos'] for info in self.data_infos]
        label2cat = {i: cat for i, cat in enumerate(self.CLASSES)}

        ret_dict = indoor_eval(
            gt_annos,
            results,
            iou_thr,
            label2cat,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d,
            NOVEL_CALSS = self.NOVEL_CLASSES)
        if show:
            self.show(results, out_dir, pipeline=pipeline)

        return ret_dict