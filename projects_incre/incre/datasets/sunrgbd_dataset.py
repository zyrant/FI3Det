# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from os import path as osp

import numpy as np

from mmdet3d.core import show_multi_modality_result, show_result_v2
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet.core import eval_map
from mmdet.datasets  import DATASETS
from mmdet3d.datasets.custom_3d  import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose
import mmcv
import warnings
from mmdet3d.core.bbox.iou_calculators import bbox_overlaps_3d
import torch
import pickle

@DATASETS.register_module()
class SUNRGBDDataset_INC(Custom3DDataset):
    r"""SUNRGBD Dataset.

    This class serves as the API for experiments on the SUNRGBD Dataset.

    See the `download page <http://rgbd.cs.princeton.edu/challenge.html>`_
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
    CLASSES = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')
    # SORTED_CLASSES = sorted(CLASSES)

    def __init__(self,
                 data_root,
                 ann_file,
                 pesudo_ann_file=None,
                 pipeline=None,
                 classes=None,
                 modality=dict(use_camera=True, use_lidar=True),
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False,
                 increment_pkl = False,
                 increament_type = 'old',
                 anno_type = '5_1_increment',
                 mode = 'alpha',
                 progressive_mode=False,
                 total_stages=None,
                 current_stage=0,
                 stage_size=[3, 2],
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
            'use_lidar' in self.modality
        assert self.modality['use_camera'] or self.modality['use_lidar']
        if mode == 'alpha':
            self.SORTED_CLASSES = sorted(self.CLASSES)
        elif mode == 'direct':
            self.SORTED_CLASSES = self.CLASSES
        elif mode == 'random':
            self.SORTED_CLASSES = ('night_stand', 'chair', 'sofa', 'bookshelf', 'desk', 'dresser', 'bathtub', 'toilet', 'bed', 'table')
        elif mode == 'count':
            self.SORTED_CLASSES = ('chair','table','desk','bed','sofa',
                              'night_stand','bookshelf','dresser','toilet','bathtub')

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
            location = annos.get('location', None)
            dimensions = annos.get('dimensions', None)
            rotation_y = annos.get('rotation_y', None)

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
                annos['gt_boxes_upright_depth'] = gt_boxes[keep_mask]
                annos['gt_num'] = int(keep_mask.sum())
                annos['name'] = gt_name[keep_mask]

                annos['location'] = location[keep_mask]
                annos['dimensions'] = dimensions[keep_mask]
                annos['rotation_y'] = rotation_y[keep_mask]
                annos['index'] = np.arange(annos['gt_num'], dtype=np.int32)

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
                - pts_filename (str, optional): Filename of point clouds.
                - file_name (str, optional): Filename of point clouds.
                - img_prefix (str, optional): Prefix of image files.
                - img_info (dict, optional): Image info.
                - calib (dict, optional): Camera calibration info.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        assert info['point_cloud']['lidar_idx'] == info['image']['image_idx']
        input_dict = dict(sample_idx=sample_idx)

        if self.modality['use_lidar']:
            pts_filename = osp.join(self.data_root, info['pts_path'])
            input_dict['pts_filename'] = pts_filename
            input_dict['file_name'] = pts_filename

        if self.modality['use_camera']:
            img_filename = osp.join(
                osp.join(self.data_root, 'sunrgbd_trainval'),
                info['image']['image_path'])
            input_dict['img_prefix'] = None
            input_dict['img_info'] = dict(filename=img_filename)
            calib = info['calib']
            rt_mat = calib['Rt']
            # follow Coord3DMode.convert_point
            rt_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]
                               ]) @ rt_mat.transpose(1, 0)
            depth2img = calib['K'] @ rt_mat
            input_dict['depth2img'] = depth2img

        if not self.test_mode:
            annos = self.get_ann_info(index)
            if self.pesudo_ann_file is not None:
                annos = self.get_pesudo_ann_info(index, annos, input_dict['pts_filename'])
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and len(annos['gt_bboxes_3d']) == 0:
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
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.int64)
            gt_names = [self.CLASSES[i] for i in gt_labels_3d]
        else:
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.int64)
            gt_names =['background']

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d, origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, gt_names = gt_names)

        if self.modality['use_camera']:
            if info['annos']['gt_num'] != 0:
                gt_bboxes_2d = info['annos']['bbox'].astype(np.float32)
            else:
                gt_bboxes_2d = np.zeros((0, 4), dtype=np.float32)
            anns_results['bboxes'] = gt_bboxes_2d
            anns_results['labels'] = gt_labels_3d

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
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.pesduo_data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = np.full((len(gt_bboxes_3d),), -1, dtype=np.int64)
            gt_names = np.array(['objects'] * len(gt_bboxes_3d))
            if 'feat_proto' in info['annos']:
                gt_feats = info['annos']['feat_proto'].astype(np.float32)
            else:
                gt_feats = np.zeros((len(gt_labels_3d), 256), dtype=np.float32)
        else:
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.int64)
            gt_names =['background']
            gt_feats = np.zeros((0, 256), dtype=np.float32)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d, origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        
        if len(gt_labels_3d) > 0:
            act_gt_bboxes_3d = annos['gt_bboxes_3d']
            act_gt_labels_3d = annos['gt_labels_3d']
            act_gt_names = annos['gt_names']
            overlaps = bbox_overlaps_3d(torch.cat((gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1), 
                                    torch.cat((act_gt_bboxes_3d.gravity_center, act_gt_bboxes_3d.tensor[:, 3:]), dim=1), 
                                    coordinate=self.temp_box_type)
            max_overlaps, _ = overlaps.max(dim=1) 

            pseudo_keep_mask = (max_overlaps < 0.25).numpy().astype(bool)
            pseudo_overlap_mask = (max_overlaps >= 0.25).numpy().astype(bool)

            new_boxes = []
            new_labels = []
            new_names = []
            new_feats = []

            # <0.25 LABEL -1
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

        
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, gt_names = gt_names, gt_feats = gt_feats)

        if self.modality['use_camera']:
            if info['annos']['gt_num'] != 0:
                gt_bboxes_2d = info['annos']['bbox'].astype(np.float32)
            else:
                gt_bboxes_2d = np.zeros((0, 4), dtype=np.float32)
            anns_results['pes_bboxes'] = gt_bboxes_2d
            anns_results['pes_labels'] = gt_labels_3d

        return anns_results
    
    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos
        # if self.filter_empty_gt and len(annos['gt_bboxes_3d']) == 0:
        #     return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example


    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        if self.modality['use_camera']:
            pipeline.insert(0, dict(type='LoadImageFromFile'))
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
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, img_metas, img = self._extract_data(
                i, pipeline, ['points', 'img_metas', 'img'])
            # scale colors to [0, 255]
            points = points.numpy()
            points[:, 3:] *= 255

            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            gt_corners = gt_bboxes.corners.numpy() if len(gt_bboxes) else None
            gt_labels = self.get_ann_info(i)['gt_labels_3d']
            pred_bboxes = result['boxes_3d']
            pred_corners = pred_bboxes.corners.numpy() if len(pred_bboxes) else None
            pred_labels = result['labels_3d']
            show_result_v2(points, gt_corners, gt_labels,
                           pred_corners, pred_labels, out_dir, file_name)

            continue  # todo: REMOVE THIS LINE

            # multi-modality visualization
            if self.modality['use_camera']:
                img = img.numpy()
                # need to transpose channel to first dim
                img = img.transpose(1, 2, 0)
                show_multi_modality_result(
                    img,
                    gt_bboxes.tensor.numpy(),
                    pred_bboxes.tensor.numpy(),
                    None,
                    out_dir,
                    file_name,
                    box_mode='depth',
                    img_metas=img_metas,
                    show=show)

    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 iou_thr_2d=(0.5, ),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: None.
            iou_thr (list[float], optional): AP IoU thresholds for 3D
                evaluation. Default: (0.25, 0.5).
            iou_thr_2d (list[float], optional): AP IoU thresholds for 2D
                evaluation. Default: (0.5, ).
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        # evaluate 3D detection performance
        if isinstance(results[0], dict):
            return self.temp_evaluate(results, metric, iou_thr, logger, show,
                                    out_dir, pipeline)
        
        # evaluate 2D detection performance
        else:
            eval_results = OrderedDict()
            annotations = [self.get_ann_info(i) for i in range(len(self))]
            iou_thr_2d = (iou_thr_2d) if isinstance(iou_thr_2d,
                                                    float) else iou_thr_2d
            for iou_thr_2d_single in iou_thr_2d:
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr_2d_single,
                    dataset=self.CLASSES,
                    logger=logger)
                eval_results['mAP_' + str(iou_thr_2d_single)] = mean_ap
            return eval_results
        
    def temp_evaluate(self,
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
        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}
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