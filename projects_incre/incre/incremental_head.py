import torch
from torch import nn
from mmdet3d.models import DETECTORS, build_detector
from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.core import bbox3d2result
from collections import defaultdict
import torch.nn.functional as F
from mmcv.ops import nms3d, nms3d_normal



@DETECTORS.register_module()
class Incremental_head(Base3DDetector):
    def __init__(self, model_cfg, pretrained=None, train_cfg=None, test_cfg=None, progressive_mode = False, total_stages=4, current_stage=0, stage_sizes = [2,2,1]):
        super(Incremental_head, self).__init__()
        self.model_cfg = model_cfg
        self.student = build_detector(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
        self.pretrained = pretrained
        self.local_iter = nn.Parameter(torch.zeros(1), requires_grad=False)
        in_channels = model_cfg['head']['in_channels']
        n_classes = model_cfg['head']['n_classes']
        base_n_class = self.student.head.num_base_classes
        self.base_n_class = base_n_class
        self.n_claases = n_classes
        self.in_channels = in_channels
        self.n_new_class = n_classes - base_n_class

        if progressive_mode:
            # self.n_new_class = (self.n_new_class // total_stages) * current_stage
            self.n_new_class =  sum(stage_sizes[:current_stage])

        self.dino_dim = model_cfg.get('dino_dim', 256)
        self.new_cls_conv_2d = nn.Parameter(torch.zeros(self.dino_dim, self.n_new_class), requires_grad=False)
        torch.nn.init.kaiming_normal_(self.new_cls_conv_2d)
        self.new_cls_conv_3d = nn.Parameter(
            self.new_cls_conv_2d.data.clone()[:self.in_channels, :], requires_grad=False)

        self.alpha_conv = nn.Linear(self.in_channels, 1)
        self.beta_conv = nn.Linear(self.dino_dim, 1)
        self.gamma_conv = nn.Linear(self.in_channels + self.dino_dim, self.n_new_class)


        self.full_id_to_novel_id = {
            self.student.head.CLASSES.index(name): i for i, name in enumerate(self.student.head.NOVEL_CLASSES)
        }
        self.base_novel_id_to_full_id = {
            i: self.student.head.CLASSES.index(name) for i, name in enumerate(self.student.head.SORTED_CLASSES)
        }

        self.novel_id_to_full_id = {
            i: self.student.head.CLASSES.index(name) for i, name in enumerate(self.student.head.NOVEL_CLASSES)
        }

        # if self.pretrained is not None:
        #     ckpt = torch.load(self.pretrained)['state_dict']
        #     self.student.load_state_dict(ckpt)
        #     print(f"[Imprinting] Loaded pretrained weights from {self.pretrained}")
        
        if self.pretrained is not None:
            ckpt = torch.load(self.pretrained, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)
            print(f"[Imprinting] Loading pretrained weights from {self.pretrained}")

            # 1. loading student 
            student_state = self.student.state_dict()
            matched_student = {}

            for k, v in state_dict.items():
                if k.startswith("student."):
                    key_wo_prefix = k[len("student."):]
                    if key_wo_prefix in student_state and student_state[key_wo_prefix].shape == v.shape:
                        matched_student[key_wo_prefix] = v
                elif k in student_state and student_state[k].shape == v.shape:
                    matched_student[k] = v

            self.student.load_state_dict(matched_student, strict=False)
            print(f"[Progressive] Loaded {len(matched_student)} params into student.")
            
            # 2. loading Incremental_head (alpha/beta/gamma）
            model_state = self.state_dict()
            matched_all = {}
            for k, v in state_dict.items():
                if k.startswith("student."):
                    continue
                if k in model_state and model_state[k].shape == v.shape:
                    matched_all[k] = v
            self.load_state_dict(matched_all, strict=False)
            print(f"[Progressive] Loaded {len(matched_all)} extra params (alpha/beta conv).")


            # 3. loading new_cls_conv_* with partial copy strategy
            for key_name, cur_param in [
                ("new_cls_conv_2d", self.new_cls_conv_2d),
                ("new_cls_conv_3d", self.new_cls_conv_3d)
            ]:
                if key_name in state_dict:
                    prev_param = state_dict[key_name]
                    num_old_classes = min(prev_param.shape[1], cur_param.shape[1])

                    with torch.no_grad():
                        cur_param[:, :num_old_classes].copy_(prev_param[:, :num_old_classes])
                     
                    print(f"[Progressive] Copied {num_old_classes} old class weights into {key_name}.")
                else:
                    print(f"[Skip] {key_name} not found in pretrained checkpoint.")

            if 'gamma_conv.weight' in state_dict:
                prev_gamma = state_dict['gamma_conv.weight']
                num_old_classes = min(prev_gamma.shape[0], self.gamma_conv.weight.shape[0])

                with torch.no_grad():
                    self.gamma_conv.weight[:num_old_classes, :].copy_(prev_gamma[:num_old_classes, :])
                    self.gamma_conv.bias[:num_old_classes].copy_(state_dict['gamma_conv.bias'][:num_old_classes])
                
                # freeze old class weights gradients
                def freeze_old_gamma_grad(grad, n=num_old_classes):
                    grad[:n, :] = 0
                    return grad

                def freeze_old_gamma_bias_grad(grad, n=num_old_classes):
                    grad[:n] = 0
                    return grad

                self.gamma_conv.weight.register_hook(freeze_old_gamma_grad)
                self.gamma_conv.bias.register_hook(freeze_old_gamma_bias_grad)
                
                print(f"[Progressive] Copied {num_old_classes} old class weights into gamma_conv.")

            # 4. freeze student
            self.student.eval()
            for p in self.student.parameters():
                p.requires_grad = False
        

        self.novel_feat_buffer_3d = defaultdict(lambda: defaultdict(list))
        self.novel_feat_buffer_2d = defaultdict(lambda: defaultdict(list))

        self.ema_momentum = 0.999
        self.ema_update_every = 1
        self.sim_thr = 0.9

    def _new_forward_single(self, x,):
        out_feat_3d = x.features
        x_2d = self.student.head.proj_head(x)

        out_feat_2d = x_2d.features
        obj_pred = self.student.head.obj_conv(x).features.sigmoid()


        norm_x_3d = F.normalize(out_feat_3d, p=2, dim=1)
        norm_x_2d = F.normalize(out_feat_2d, p=2, dim=1)

        weight_2d = F.normalize(self.new_cls_conv_2d, p=2, dim=0)
        weight_3d = F.normalize(self.new_cls_conv_3d, p=2, dim=0)

        cls_pred_2d = (torch.matmul(norm_x_2d, weight_2d)+1) / 2
        cls_pred_3d = (torch.matmul(norm_x_3d, weight_3d)+1) / 2

        alpha = self.alpha_conv(norm_x_3d) # N, C
        beta = self.beta_conv(norm_x_2d) # N, C
        

        cat = torch.cat((alpha, beta), dim=1)
        cat = torch.softmax(cat, dim=1)
        alpha, beta = cat[:, 0:1], cat[:, 1:2]
        gamma = self.gamma_conv(torch.cat((norm_x_3d, norm_x_2d), dim=1)) # N, C
        gamma = gamma.softmax(dim=1) if gamma.size(1) > 1 else gamma.sigmoid()

        
        
        cls_preds_3d, cls_preds_2d, points, cls_feats_3d, cls_feats_2d, new_cls_preds = [], [], [], [], [], []
        for i, permutation in enumerate(x.decomposition_permutations):
            points.append(x.coordinates[permutation][:, 1:] * self.student.head.voxel_size)

            cls_preds_3d.append(cls_pred_3d[permutation])
            cls_preds_2d.append(cls_pred_2d[permutation])

            cls_feats_3d.append(norm_x_3d[permutation])
            cls_feats_2d.append(norm_x_2d[permutation])

            scores_new = alpha[permutation] * cls_pred_3d[permutation] + beta[permutation] * cls_pred_2d[permutation]
            scores_new = scores_new * gamma[permutation]
            # scores_new = cls_pred_3d[permutation]
            new_cls_preds.append(scores_new)
            
        return cls_preds_3d, cls_preds_2d, points, cls_feats_3d, cls_feats_2d, new_cls_preds
    
    def new_forward(self, x):

        oldbbox_preds, cls_preds, points, out_feats, _, obj_preds = self.student.head.forward(x)
        cls_preds_3d, cls_preds_2d, points, cls_feats_3d, cls_feats_2d, new_cls_preds = [], [], [], [], [], []
        for i in range(len(x)):
            cls_pred_3d, cls_pred_2d, point, cls_feat_3d, cls_feat_2d, new_cls_pred = self._new_forward_single(x[i])
            cls_preds_3d.append(cls_pred_3d)
            cls_preds_2d.append(cls_pred_2d)
            points.append(point)
            cls_feats_3d.append(cls_feat_3d)
            cls_feats_2d.append(cls_feat_2d)
            new_cls_preds.append(new_cls_pred)

        return cls_preds_3d, cls_preds_2d, points, cls_feats_3d, cls_feats_2d, new_cls_preds, obj_preds, oldbbox_preds

    def _update_weight(self, cls_preds_3d, cls_preds_2d, points, cls_feats_3d, cls_feats_2d, new_cls_preds, obj_preds, oldbbox_preds,
              gt_bboxes, gt_labels, img_metas, scene_points):
        
        cls_losses, pos_masks = [], []

        for i in range(len(img_metas)):
            cls_loss, pos_mask = self._update_weight_single(
                cls_preds_3d=[x[i] for x in cls_preds_3d],
                cls_preds_2d=[x[i] for x in cls_preds_2d],
                points=[x[i] for x in points],
                cls_feats_3d=[x[i] for x in cls_feats_3d],
                cls_feats_2d = [x[i] for x in cls_feats_2d],
                new_cls_preds = [x[i] for x in new_cls_preds],
                obj_preds = [x[i] for x in obj_preds],
                oldbbox_preds = [x[i] for x in oldbbox_preds],
                img_meta=img_metas[i],
                support_bboxes_3d=gt_bboxes[i],
                support_labels_3d=gt_labels[i],
                scene_points = scene_points[i])
            cls_losses.append(cls_loss)
            pos_masks.append(pos_mask)

        cls_loss=torch.mean(torch.stack(cls_losses))

        return dict(cls_loss = cls_loss)


    # per scene
    def _update_weight_single(self,
                     cls_preds_3d, 
                     cls_preds_2d, 
                     points, 
                     cls_feats_3d, 
                     cls_feats_2d,
                     new_cls_preds,
                     obj_preds,
                     oldbbox_preds, 
                     support_bboxes_3d,
                     support_labels_3d,
                     img_meta,
                     scene_points):
        
  
        assigned_ids = self.student.head.assigner.assign(points, support_bboxes_3d, support_labels_3d, img_meta)
        cls_preds_3d = torch.cat(cls_preds_3d)
        cls_preds_2d = torch.cat(cls_preds_2d)
        cls_feats_3d = torch.cat(cls_feats_3d)
        cls_feats_2d = torch.cat(cls_feats_2d)
        points = torch.cat(points)
        obj_preds = torch.cat(obj_preds).sigmoid()
        new_cls_preds = torch.cat(new_cls_preds)
        oldbbox_preds = torch.cat(oldbbox_preds)
        
        n_classes = cls_preds_3d.shape[1]
        sample_idx = img_meta['sample_idx']
        pos_mask = assigned_ids >= 0

        new_gt_labels = torch.tensor(
                [self.full_id_to_novel_id[int(i)] for i in support_labels_3d],
                dtype=support_labels_3d.dtype,
                device=support_labels_3d.device
            )
        cls_targets = torch.where(pos_mask, new_gt_labels[assigned_ids], -1)
        select_new_cls_preds = new_cls_preds[pos_mask]
        select_cls_target = cls_targets[pos_mask]

        target_onehot = torch.zeros_like(select_new_cls_preds)   # [N_pos, C]
        target_onehot.scatter_(1, select_cls_target.unsqueeze(1), 1)

        pos_loss = (1 - select_new_cls_preds) * target_onehot
        neg_loss = select_new_cls_preds * (1 - target_onehot)
        cls_loss = pos_loss + neg_loss
        cls_loss = cls_loss.mean()
        neg_loss_v2 = new_cls_preds[~pos_mask].mean()
        cls_loss = cls_loss + neg_loss_v2

        unique_class_ids = cls_targets[pos_mask].unique()

        for class_id in unique_class_ids:
            selected_masks = cls_targets == class_id
            self_cls_feats_3d = cls_feats_3d[selected_masks]
            self_cls_feats_2d = cls_feats_2d[selected_masks]

            if self_cls_feats_3d.size(0) == 0:
                continue

            class_id = int(class_id)
            sample_idx = str(img_meta["sample_idx"])
            new_feat_3d = self_cls_feats_3d.mean(dim=0, keepdim=True)  # [1, C]
            new_feat_2d = self_cls_feats_2d.mean(dim=0, keepdim=True)  # [1, C]

            self.update_novel_feat_buffer(self.novel_feat_buffer_3d, class_id, sample_idx, new_feat_3d, sim_thr=self.sim_thr)
            self.update_novel_feat_buffer(self.novel_feat_buffer_2d, class_id, sample_idx, new_feat_2d, sim_thr=self.sim_thr)

        self.local_iter += 1.


        return cls_loss, pos_mask   

    
    def update_novel_feat_buffer(self, novel_feat_buffer, class_id, sample_idx,
                                new_feat, sim_thr=0.7):
    
        scene_feat_list = novel_feat_buffer[class_id][sample_idx]  # list of [M_i, C]

        if len(scene_feat_list) > 0:
            existing_feats = torch.cat(scene_feat_list, dim=0)  # [M_total, C]

            if new_feat.shape[0] != scene_feat_list[0].shape[0]:
                return  

        novel_feat_buffer[class_id][sample_idx].append(new_feat)


    def collect_feats_for_imprinting(self, scene_points, gt_bboxes_3d, gt_labels_3d, img_metas):
        self.student.eval()
        # with torch.no_grad():
        x = self.student.extract_feats(scene_points)
        cls_preds_3d, cls_preds_2d, points, cls_feats_3d, cls_feats_2d, new_cls_preds, obj_preds, oldbbox_preds = self.new_forward(x)
        loss_dict = self._update_weight(cls_preds_3d, cls_preds_2d, points, cls_feats_3d, cls_feats_2d, new_cls_preds, obj_preds, oldbbox_preds,
                            gt_bboxes_3d, gt_labels_3d, img_metas, scene_points=scene_points)

        self.update_novel_class_weights(self.novel_feat_buffer_3d, self.new_cls_conv_3d, self.local_iter, ema_momentum = self.ema_momentum)
        self.update_novel_class_weights(self.novel_feat_buffer_2d, self.new_cls_conv_2d, self.local_iter, ema_momentum = self.ema_momentum)

        return loss_dict
    
    @torch.no_grad()
    def update_novel_class_weights(self, novel_feat_buffer, new_cls_conv,
                               local_iter, ema_momentum=0.9):
      
        for novel_id, scene_dict in novel_feat_buffer.items():
            per_scene_means = []
            for scene_id, feats in scene_dict.items():
                if feats:  
                    scene_mean = torch.stack(feats).mean(dim=0)  
                    per_scene_means.append(scene_mean)

            if not per_scene_means:  
                continue

            mean_feat = torch.cat(per_scene_means).mean(dim=0)  
            mean_feat = F.normalize(mean_feat, p=2, dim=0)

            current_feat = new_cls_conv[:, novel_id]

            momentum = min(1 - 1 / ((local_iter / 10) + 1), ema_momentum)
            updated_feat = F.normalize(
                momentum * current_feat + (1 - momentum) * mean_feat,
                p=2, dim=0
            )

            new_cls_conv.data[:, novel_id] = updated_feat

 

    def forward_train(self, points, gt_bboxes_3d, gt_labels_3d, img_metas, **kwargs):
        loss_dict = self.collect_feats_for_imprinting(points, gt_bboxes_3d, gt_labels_3d, img_metas)
        dummy_loss = torch.zeros(1, device=points[0].device, requires_grad=True).sum()
        loss_dict['cls_loss'] = loss_dict['cls_loss'] + dummy_loss
        
        return loss_dict
        
    def _get_bboxes_single(self, bbox_preds, cls_preds, cls_preds_3d, cls_preds_2d, new_cls_preds, points, obj_preds, img_meta, scene_points):
        scores_base = torch.cat(cls_preds).sigmoid() 
        scores_new = torch.cat(new_cls_preds)  * torch.cat(obj_preds).sigmoid()   

        scores = torch.cat((scores_base, scores_new), dim=1)
        bbox_preds = torch.cat(bbox_preds)
        orgin_points = torch.cat(points)
        points = torch.cat(points)
        max_scores_base, _ = scores_base.max(dim=1)
        max_scores_new, _ = scores_new.max(dim=1)
        back_flag = 1 - max_scores_base
        
        if len(max_scores_base) > self.student.head.test_cfg.nms_pre > 0:
            base, ids_base = max_scores_base.topk(self.student.head.test_cfg.nms_pre)
            new, ids_new = max_scores_new.topk(self.student.head.test_cfg.nms_pre)
            mask = ~torch.isin(ids_new, ids_base)
            filtered_ids_new = ids_new[mask]

            final_ids = torch.cat([ids_base, filtered_ids_new], dim=0)
            bbox_preds = bbox_preds[final_ids]
            scores = scores[final_ids]
            points = points[final_ids]

        boxes = self.student.head._bbox_pred_to_bbox(points, bbox_preds)
        boxes, scores, labels = self.student.head._nms(boxes, scores, img_meta)
        new_labels = torch.tensor(
                [self.base_novel_id_to_full_id[int(i)] for i in labels],
                dtype=labels.dtype,
                device=labels.device
            )
        
        return boxes, scores, new_labels

        
    def _get_bboxes(self, bbox_preds, cls_preds, cls_preds_3d, cls_preds_2d, new_cls_preds, points, obj_preds, img_metas, scene_points):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                cls_preds_3d = [x[i] for x in cls_preds_3d],
                cls_preds_2d = [x[i] for x in cls_preds_2d],
                new_cls_preds = [x[i] for x in new_cls_preds],
                points=[x[i] for x in points],
                obj_preds = [x[i] for x in obj_preds],
                img_meta=img_metas[i],
                scene_points = scene_points[i] )
            results.append(result)
        return results

    def simple_test(self, scene_points, img_metas, *args, **kwargs):

        x = self.student.extract_feats(scene_points)

        bbox_preds, cls_preds, points, out_feats, _, obj_preds = self.student.head.forward(x)
        cls_preds_3d, cls_preds_2d, points, cls_feats_3d, cls_feats_2d, new_cls_preds, obj_preds, out_feats = self.new_forward(x)

        bbox_list = self._get_bboxes(bbox_preds, cls_preds, cls_preds_3d, cls_preds_2d, new_cls_preds, points, obj_preds, img_metas, scene_points)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
    
    def aug_test(self, points, img_metas, **kwargs):
        assert NotImplementedError, "aug test not implemented"
        # TODO: [c7w] aug_test
        pass


    def extract_feat(self, points, img_metas):
        assert NotImplementedError, "cannot directly use extract_feat in ensembled model"
        pass

    def forward_test(self, points, img_metas, img=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(points, 'points'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(points)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(points), len(img_metas)))

        if num_augs == 1:
            img = [img] if img is None else img
            return self.simple_test(points[0], img_metas[0], img[0], **kwargs)
        else:
            return self.aug_test(points, img_metas, img, **kwargs)
    
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
        thr = thr if thr is not None else self.student.head.test_cfg.score_thr
        iou_thr = iou_thr if iou_thr is not None else self.student.head.test_cfg.iou_thr
        for i in range(n_classes):
            ids = scores[:, i] > thr[i]
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
