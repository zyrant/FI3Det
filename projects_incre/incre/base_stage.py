try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

import torch
from mmcv.cnn import bias_init_with_prob
from mmcv.ops import nms3d, nms3d_normal
from mmcv.runner import BaseModule
from torch import nn

from mmdet3d.models.builder import HEADS, build_loss
from mmdet.core.bbox.builder import BBOX_ASSIGNERS, build_assigner
import torch.nn.functional as F
from mmdet.core import reduce_mean
import copy
from torch_scatter import scatter_mean, scatter_add
import math
from .utils import show_result, get_face_distances, get_centerness, get_gaussian_center_weight, dice_loss, extract_roi_single


@HEADS.register_module()
class Base_stage(BaseModule):
    def __init__(self,
                 class_names,
                 anno_type,
                 n_classes,
                 in_channels,
                 n_reg_outs,
                 voxel_size,
                 assigner,
                 bbox_loss=dict(type='AxisAlignedIoULoss', reduction='none'),
                 cls_loss=dict(type='FocalLoss', reduction='none'),
                 train_cfg=None,
                 test_cfg=None,
                 box_uncertain_mode = 'box',
                 mode = 'alpha',
                 inc_class_names = None,

                 ):
        
        super(Base_stage, self).__init__()
        self.voxel_size = voxel_size
        self.assigner = build_assigner(assigner)
        self.bbox_loss = build_loss(bbox_loss)
        temp_bbox_loss = copy.deepcopy(bbox_loss)
        temp_bbox_loss['reduction'] = 'mean'
        self.temp_bbox_loss = build_loss(temp_bbox_loss)
        self.cls_loss = build_loss(cls_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.n_reg_outs = n_reg_outs
        self.CLASSES = class_names
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.box_uncertain_mode = box_uncertain_mode
        
        if mode == 'alpha':
            self.SORTED_CLASSES = sorted(self.CLASSES)
        elif mode == 'random':
            assert inc_class_names is not None, "inc_class_names should be provided when mode is 'random'"
            self.SORTED_CLASSES = inc_class_names
        elif mode == 'direct':
            self.SORTED_CLASSES = self.CLASSES
        elif mode == 'count':
            assert inc_class_names is not None, "inc_class_names should be provided when mode is 'count'"
            self.SORTED_CLASSES = inc_class_names
        else:
            raise ImportError('No such mode: {}'.format(mode))
            
        self.num_novel_classes = int(anno_type.split('_')[0])
        self.num_base_classes = len(self.CLASSES) - self.num_novel_classes
        self.BASE_CLASSES = self.SORTED_CLASSES[:self.num_base_classes]
        self.NOVEL_CLASSES = self.SORTED_CLASSES[self.num_base_classes:]

        # full_id -> base_id
        self.full_id_to_base_id = {
            self.CLASSES.index(name): i for i, name in enumerate(self.BASE_CLASSES)
        }

        # base_id -> full_id
        self.base_id_to_full_id = {
            i: self.CLASSES.index(name) for i, name in enumerate(self.BASE_CLASSES)
        }

        
        self.dino_channels = 256 # 256
        self.in_channels = in_channels
        self.local_iter = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.proj_head =  nn.Sequential(
            ME.MinkowskiConvolution(in_channels, self.dino_channels, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(self.dino_channels),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(self.dino_channels, self.dino_channels, kernel_size=1, dimension=3))
        
        self._init_layers(n_classes, self.num_base_classes, in_channels, n_reg_outs)


    def _init_layers(self, n_classes, base_n_class, in_channels, n_reg_outs):
        self.bbox_conv = ME.MinkowskiConvolution(
            in_channels, n_reg_outs, kernel_size=1, bias=True, dimension=3)

        self.cls_conv = ME.MinkowskiConvolution(
            in_channels, base_n_class, kernel_size=1, bias=True, dimension=3)
        self.obj_conv = ME.MinkowskiConvolution(
            in_channels, 1, kernel_size=1, bias=True, dimension=3)

    def init_weights(self):
        nn.init.normal_(self.bbox_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))
      

    # per level
    def _forward_single(self, x):
        reg_final = self.bbox_conv(x).features
        reg_distance = torch.exp(reg_final[:, 3:6])
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_final[:, :3], reg_distance, reg_angle), dim=1)

        cls_pred = self.cls_conv(x).features
        obj_pred = self.obj_conv(x).features
        out_feat_2d = self.proj_head(x).features
        

        bbox_preds, cls_preds, points, out_feats_2d, out_feats_3d, obj_preds = [], [], [], [], [], []
        for permutation in x.decomposition_permutations:
            bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            points.append(x.coordinates[permutation][:, 1:] * self.voxel_size)

            obj_preds.append(obj_pred[permutation])
            out_feats_2d.append(out_feat_2d[permutation])
            out_feats_3d.append(x.features[permutation])

        return bbox_preds, cls_preds, points, out_feats_2d, out_feats_3d, obj_preds

    def forward(self, x):

        bbox_preds, cls_preds, points, out_feats_2d, out_feats_3d, obj_preds = [], [], [], [], [], []
        for i in range(len(x)):
            bbox_pred, cls_pred, point, out_feat_2d, out_feat_3d, obj_pred = self._forward_single(x[i])
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            points.append(point)
            out_feats_2d.append(out_feat_2d)
            out_feats_3d.append(out_feat_3d)
            obj_preds.append(obj_pred)            
        
        return bbox_preds, cls_preds, points, out_feats_2d, out_feats_3d, obj_preds

    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.
        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0]
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 3],
            bbox_pred[:, 4],
            bbox_pred[:, 5]], -1)

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox

        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 3] + bbox_pred[:, 4]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)
    
    @torch.no_grad()
    def compute_box_var_weight(
        self,
        out_feats_2d: torch.Tensor,       
        out_feats_3d: torch.Tensor,   
        min_inds: torch.Tensor,           
        unkown_pos_mask: torch.Tensor,    
        pos_sdf_targets: torch.Tensor,    
        n_unknow_boxes: int,              
        tau: float = 0.5,
        # robust_clip: float = 0.05,
        l2norm: bool = True
    ):
        K = int(n_unknow_boxes)
        if K == 0 or unkown_pos_mask.sum() == 0:
            return out_feats_2d.new_zeros(0), torch.tensor(1.0, device=out_feats_2d.device)

        feats_m_2d = out_feats_2d[unkown_pos_mask]     
        if l2norm:
            feats_m_2d = F.normalize(feats_m_2d, dim=1)
        # feats_m_3d = out_feats_3d[unkown_pos_mask]     
        # if l2norm:
        #     feats_m_3d = F.normalize(feats_m_3d, dim=1)

        feats_m = feats_m_2d
        M, D = feats_m.shape

        box_id_per_point = min_inds[unkown_pos_mask].long()    

        one = feats_m.new_ones(M)
        cnt = scatter_add(one, box_id_per_point, dim=0, dim_size=K).clamp_min(1.0)                     # [K]
        sum_f = scatter_add(feats_m, box_id_per_point.unsqueeze(-1).expand(-1, D), dim=0, dim_size=K)  # [K,D]
        sum_sq = scatter_add(feats_m.pow(2), box_id_per_point.unsqueeze(-1).expand(-1, D), dim=0, dim_size=K)  # [K,D]

        mean = sum_f / cnt.unsqueeze(-1)             
        E_sq = sum_sq / cnt.unsqueeze(-1)            
        var_vec = (E_sq - mean.pow(2)).clamp_min(0.) 
        var_box = (var_vec.sum(dim=-1) / (D + 1e-6)) 

        # if robust_clip > 0 and (K > 4):
        #     lo = torch.quantile(var_box, robust_clip)
        #     hi = torch.quantile(var_box, 1 - robust_clip)
        #     var_box = var_box.clamp(lo.item(), hi.item())

        fc_box = torch.exp(-(var_box / (tau + 1e-6))).clamp(0., 1.)   
        fc_weight_m = fc_box[box_id_per_point].clamp(0., 1.)          

        weight = (fc_weight_m * pos_sdf_targets.squeeze(-1)).clamp_min(1e-6)
        denorm = max(reduce_mean(weight.sum().detach()), 1e-6)
        return weight, denorm

    @torch.no_grad()
    def compute_box_weight(
        self,
        out_feats_2d: torch.Tensor,   
        min_inds: torch.Tensor,      
        unkown_pos_mask: torch.Tensor,
        pos_gaussion_targets: torch.Tensor,
        boxes: torch.Tensor,         
        tau: float = 0.5,
        l2norm: bool = True
    ):
        K = boxes.shape[0]
        if K == 0 or unkown_pos_mask.sum() == 0:
            return out_feats_2d.new_zeros(0), torch.tensor(1.0, device=out_feats_2d.device)

        feats_m = out_feats_2d[unkown_pos_mask]
        if l2norm:
            feats_m = F.normalize(feats_m, dim=1)
        M, D = feats_m.shape
        box_id_per_point = min_inds[unkown_pos_mask].long()  # [M]

        sum_f = scatter_add(feats_m, box_id_per_point.unsqueeze(-1).expand(-1, D), dim=0, dim_size=K)  # [K,D]
        cnt = scatter_add(torch.ones_like(box_id_per_point, dtype=torch.float, device=feats_m.device),
                        box_id_per_point, dim=0, dim_size=K).clamp_min(1.0).unsqueeze(-1)               # [K,1]
        mean_dir = sum_f / cnt  # [K,D]

        consistency = mean_dir.norm(dim=-1).clamp(0., 1.)  # [K], -> 1 better

        box_weight = consistency.clamp(0., 1.)
        weight_m = box_weight[box_id_per_point]

        weight = (weight_m * pos_gaussion_targets.squeeze(-1)).clamp_min(1e-6)
        denorm = max(reduce_mean(weight.sum().detach()), 1e-6)
        return weight, denorm



    # per scene
    def _loss_single(self,
                     bbox_preds, 
                     cls_preds, 
                     points, 
                     out_feats_2d, 
                     out_feats_3d,
                     obj_preds,
                     gt_bboxes,
                     gt_labels,
                     gt_feats,
                     img_meta,
                     scene_points):
        
        gt_mask = gt_labels >=0
        pseudo_mask = gt_labels == -1
        quality_mask = gt_labels == -2

        pseudo_gt_bboxes = gt_bboxes[pseudo_mask]
        pseudo_gt_labels = gt_labels[pseudo_mask]

        quality_gt_bboxes = gt_bboxes[quality_mask]
        quality_gt_labels = gt_labels[quality_mask]

        gt_bboxes = gt_bboxes[gt_mask]
        gt_labels = gt_labels[gt_mask]
        gt_feats = gt_feats[pseudo_mask]

        assigned_ids = self.assigner.assign(points, gt_bboxes, gt_labels, img_meta)
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)
        obj_preds = torch.cat(obj_preds)
        out_feats_2d = torch.cat(out_feats_2d)
        out_feats_3d = torch.cat(out_feats_3d)

        # cls loss
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0

        if len(gt_labels) > 0:
            new_gt_labels = torch.tensor(
                [self.full_id_to_base_id[int(i)] for i in gt_labels],
                dtype=gt_labels.dtype,
                device=gt_labels.device
            )
            cls_targets = torch.where(pos_mask, new_gt_labels[assigned_ids], -1)
        else:
            cls_targets = gt_labels.new_full((len(pos_mask),), -1)
        cls_loss = self.cls_loss(cls_preds, cls_targets)


        # bbox loss
        pos_bbox_preds = bbox_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            bbox_targets = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(points.device)
            pos_bbox_targets = bbox_targets[assigned_ids][pos_mask]
            if pos_bbox_preds.shape[1] == 6:
                pos_bbox_targets = pos_bbox_targets[:, :6]
            bbox_loss = self.bbox_loss(
                self._bbox_to_loss(
                    self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
                self._bbox_to_loss(pos_bbox_targets))
        else:
            bbox_loss = None

        
        # -----unknown part ---
        pos_targets = gt_bboxes.points_in_boxes_all(points).sum(dim=-1) > 0  
        back_targets = (~pos_mask) & (~pos_targets)  

        n_points = len(points)
        n_unknow_boxes = len(pseudo_gt_bboxes)

        obj_targets = points.new_zeros(n_points)

        unknow_bbox_loss = None
        unknow_feat_loss = None

        if n_unknow_boxes > 0:

            unknow_box_3d = torch.cat(
                (pseudo_gt_bboxes.gravity_center, pseudo_gt_bboxes.tensor[:, 3:]), dim=1
            ).to(points.device)                                  # [K,7] / [K,6]
            n_unk = unknow_box_3d.shape[0]


            unknow_box = unknow_box_3d.expand(n_points, n_unk, unknow_box_3d.shape[-1])
            points_expand = points.unsqueeze(1).expand(n_points, n_unk, 3)

            face_distances = get_face_distances(points_expand, unknow_box)    
            inside_box_condition = face_distances.min(dim=-1).values > 0      

            gaussion_weight = get_gaussian_center_weight(face_distances)          
            gaussion_weight = torch.where(inside_box_condition, gaussion_weight,
                                    torch.zeros_like(gaussion_weight))

            volumes = pseudo_gt_bboxes.volume.unsqueeze(0).expand(n_points, n_unk).to(points.device)
            volumes = torch.where(inside_box_condition, volumes, points.new_tensor(1e8))
            min_volumes, min_inds = volumes.min(dim=1)                        

            gaussion_targets = gaussion_weight[torch.arange(n_points), min_inds]  

            unkown_pos_mask = (min_volumes != 1e8) & back_targets              
            unknow_bbox_targets = unknow_box[torch.arange(n_points), min_inds] 
            if bbox_preds.shape[1] == 6:
                unknow_bbox_targets = unknow_bbox_targets[:, :6]

            if unkown_pos_mask.any():
                unknow_pos_points = points[unkown_pos_mask]
                unknow_pos_bbox_preds = bbox_preds[unkown_pos_mask]
                unknow_pos_bbox_targets = unknow_bbox_targets[unkown_pos_mask]
                pos_gaussion_targets = gaussion_targets[unkown_pos_mask].unsqueeze(1)  

                gt_feats_full = gt_feats.expand(n_points, n_unk, self.dino_channels + 2)
                gt_feats_pick = gt_feats_full[torch.arange(n_points), min_inds]       
                gt_feats_pick = gt_feats_pick[unkown_pos_mask]
                gt_feats_pick = gt_feats_pick[:, :self.dino_channels]                  

                unknow_pos_bbox_preds = self._bbox_pred_to_bbox(unknow_pos_points, unknow_pos_bbox_preds)

                # ------ weight------
                if self.box_uncertain_mode == 'var':
                    weight, denorm = self.compute_box_var_weight(
                        out_feats_2d.clone().detach(), out_feats_3d.clone().detach(), min_inds, unkown_pos_mask, pos_gaussion_targets, n_unknow_boxes
                    )
                elif self.box_uncertain_mode == 'box':
                    b3d = unknow_box_3d
                    if bbox_preds.shape[1] == 6:
                        b3d = b3d[:, :6]
                    weight, denorm = self.compute_box_weight(
                        out_feats_2d.clone().detach(), min_inds, unkown_pos_mask, pos_gaussion_targets, b3d
                    )
                else:
                    weight = pos_gaussion_targets.squeeze(-1)
                    denorm = max(reduce_mean(weight.sum().detach()), 1e-6)

                # box loss（soft weight）
                unknow_bbox_loss = self.temp_bbox_loss(
                    self._bbox_to_loss(unknow_pos_bbox_preds),
                    self._bbox_to_loss(unknow_pos_bbox_targets),
                    weight=weight,
                    avg_factor=denorm
                    )

                unknow_feats_2d = out_feats_2d[unkown_pos_mask]
                unknow_feats_2d = F.normalize(unknow_feats_2d, dim=1)
                cos_sim = F.cosine_similarity(unknow_feats_2d, gt_feats_pick, dim=1)
                unknow_feat_loss = ((1.0 - cos_sim) * weight).sum() / denorm

                # ------- obj_targets -------
                obj_targets[unkown_pos_mask] = weight  

        # -------- obj branch--------
        obj_loss_bce = F.binary_cross_entropy_with_logits(
            obj_preds.squeeze(-1)[back_targets],
            obj_targets.float()[back_targets],
        )
        obj_loss_dice = dice_loss(
            obj_preds.squeeze(-1)[back_targets],
            obj_targets.float()[back_targets],
        )

        return bbox_loss, cls_loss, pos_mask, obj_loss_bce, obj_loss_dice, unknow_bbox_loss, unknow_feat_loss

    def _loss(self, bbox_preds, cls_preds, points, out_feats_2d, out_feats_3d, obj_preds,
              gt_bboxes, gt_labels, gt_feats, img_metas, scene_points):
        
        bbox_losses, cls_losses, pos_masks, obj_losses_bce, obj_losses_dice, unknow_bbox_losses, unknow_feat_losses = [], [], [],[], [], [], []
        for i in range(len(img_metas)):
            bbox_loss, cls_loss, pos_mask, obj_loss_bce, obj_loss_dice, unknow_bbox_loss, unknow_feat_loss  = self._loss_single(
            bbox_preds=[x[i] for x in bbox_preds],
            cls_preds=[x[i] for x in cls_preds],
            points=[x[i] for x in points],
            out_feats_2d =[x[i] for x in out_feats_2d],
            out_feats_3d = [x[i] for x in out_feats_3d],
            obj_preds = [x[i] for x in obj_preds],
            img_meta=img_metas[i],
            gt_bboxes=gt_bboxes[i],
            gt_labels=gt_labels[i],
            gt_feats = gt_feats[i],
            scene_points = scene_points[i])

            if bbox_loss is not None:
                bbox_losses.append(bbox_loss)

            cls_losses.append(cls_loss)
            pos_masks.append(pos_mask)

            if unknow_bbox_loss is not None:
                unknow_bbox_losses.append(unknow_bbox_loss)
                unknow_feat_losses.append(unknow_feat_loss)

                obj_losses_bce.append(obj_loss_bce)
                obj_losses_dice.append(obj_loss_dice)

        bbox_loss=torch.mean(torch.cat(bbox_losses))
        cls_loss=torch.sum(torch.cat(cls_losses)) / torch.sum(torch.cat(pos_masks))
        obj_loss_dice=torch.mean(torch.stack(obj_losses_dice))
        obj_loss_bce=torch.mean(torch.stack(obj_losses_bce))
        unknow_bbox_loss=torch.mean(torch.stack(unknow_bbox_losses))
        unknow_feat_loss=torch.mean(torch.stack(unknow_feat_losses))

        return dict(
            bbox_loss=bbox_loss,
            cls_loss=cls_loss,
            obj_loss_bce = obj_loss_bce * 0.25,
            obj_loss_dice = obj_loss_dice * 0.25,
            unknow_feat_loss = unknow_feat_loss,
            unknow_bbox_loss = unknow_bbox_loss * 0.25
            )
    

    def forward_train(self, x, gt_bboxes, gt_labels, gt_feats, img_metas, scene_points):
        bbox_preds, cls_preds, points, out_feats_2d, out_feats_3d, obj_preds = self(x)
        self.local_iter += 1
        return self._loss(bbox_preds, cls_preds, points, out_feats_2d, out_feats_3d, obj_preds,
                          gt_bboxes, gt_labels, gt_feats, img_metas, scene_points)

    def _nms(self, bboxes, scores, img_meta, thr=None, iou_thr=None):
        """Multi-class nms for a single scene.
        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.
        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        thr = thr if thr is not None else self.test_cfg.score_thr
        iou_thr = iou_thr if iou_thr is not None else self.test_cfg.iou_thr
        for i in range(n_classes):
            ids = scores[:, i] > thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels

    def _get_bboxes_single(self, bbox_preds, cls_preds, points, img_meta):
        scores = torch.cat(cls_preds).sigmoid()
        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)

        if len(scores) > self.test_cfg.nms_pre > 0:
            _, ids = max_scores.topk(self.test_cfg.nms_pre)
            bbox_preds = bbox_preds[ids]
            scores = scores[ids]
            points = points[ids]

        boxes = self._bbox_pred_to_bbox(points, bbox_preds)
        boxes, scores, labels = self._nms(boxes, scores, img_meta)
        new_labels = torch.tensor(
                [self.base_id_to_full_id[int(i)] for i in labels],
                dtype=labels.dtype,
                device=labels.device
            )
        return boxes, scores, new_labels

    def _get_bboxes(self, bbox_preds, cls_preds, points, img_metas):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i])
            results.append(result)
        return results

    def forward_test(self, x, img_metas):
        bbox_preds, cls_preds, points, _, _, obj_preds = self(x)
        return self._get_bboxes(bbox_preds, cls_preds, points, img_metas)



