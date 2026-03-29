import torch
from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from matplotlib import pyplot as plt
import mmcv
import os.path as osp
import numpy as np
import torch_scatter

def get_face_distances(points, boxes):
    """Calculate distances from point to box faces.

    Args:
        points (Tensor): Final locations of shape (N_points, N_boxes, 3).
        boxes (Tensor): 3D boxes of shape (N_points, N_boxes, 7)

    Returns:
        Tensor: Face distances of shape (N_points, N_boxes, 6),
            (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).
    """
    shift = torch.stack(
        (points[..., 0] - boxes[..., 0], points[..., 1] - boxes[..., 1],
            points[..., 2] - boxes[..., 2]),
        dim=-1).permute(1, 0, 2)
    shift = rotation_3d_in_axis(
        shift, -boxes[0, :, 6], axis=2).permute(1, 0, 2)
    centers = boxes[..., :3] + shift
    dx_min = centers[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
    dx_max = boxes[..., 0] + boxes[..., 3] / 2 - centers[..., 0]
    dy_min = centers[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
    dy_max = boxes[..., 1] + boxes[..., 4] / 2 - centers[..., 1]
    dz_min = centers[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
    dz_max = boxes[..., 2] + boxes[..., 5] / 2 - centers[..., 2]
    return torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max),
                        dim=-1)


def get_centerness(face_distances):
    """Compute point centerness w.r.t containing box.

    Args:
        face_distances (Tensor): Face distances of shape (B, N, 6),
            (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).

    Returns:
        Tensor: Centerness of shape (B, N).
    """
    x_dims = face_distances[..., [0, 1]]
    y_dims = face_distances[..., [2, 3]]
    z_dims = face_distances[..., [4, 5]]
    centerness_targets = x_dims.min(dim=-1)[0] / x_dims.max(dim=-1)[0] * \
        y_dims.min(dim=-1)[0] / y_dims.max(dim=-1)[0] * \
        z_dims.min(dim=-1)[0] / z_dims.max(dim=-1)[0]
    
    return torch.pow(centerness_targets, 1 / 3)

def get_gaussian_center_weight(
    face_distances: torch.Tensor,
    sigma: float = 0.5,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Gaussian weight based on distance to the BOX CENTER (center bright, edge dark).

    Args:
        face_distances (Tensor): (N_pts, N_boxes, 6) = (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).
        boxes (Tensor): (N_pts, N_boxes, 7), where (..., 3:6) are (w, l, h).
        sigma (float): Gaussian sigma. Smaller = sharper peak at center.
        eps (float): numerical epsilon.

    Returns:
        Tensor: (N_pts, N_boxes) in [0, 1], center≈1, edges≈0
    """
    
    a_x = face_distances[..., 0]
    b_x = face_distances[..., 1]
    a_y = face_distances[..., 2]
    b_y = face_distances[..., 3]
    a_z = face_distances[..., 4]
    b_z = face_distances[..., 5]

    # normalized center distance per axis: |b-a| / (a+b)
    tx = (b_x - a_x).abs() / (a_x + b_x + eps)
    ty = (b_y - a_y).abs() / (a_y + b_y + eps)
    tz = (b_z - a_z).abs() / (a_z + b_z + eps)

    r2 = tx**2 + ty**2 + tz**2
    weights = torch.exp(-r2 / (2.0 * sigma * sigma))
    weights = weights.clamp_(0.0, 1.0)

    return weights

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)  
    return loss.mean()

# TD3D
def extract_roi_single(coordinates, features, min_pts_threshold, rois):
    # coordinates: of shape (n_points, 3)
    # features: of shape (n_points, c)
    # voxel_size: float
    # rois: of shape (n_rois, 7)
    # -> new indices of shape n_new_points
    # -> new coordinates of shape (n_new_points, 3)
    # -> new features of shape (n_new_points, c + 3)
    # -> new rois of shape (n_new_rois, 7)
    # -> new scores of shape (n_new_rois)
    # -> new labels of shape (n_new_rois)
    n_points = len(coordinates)
    n_boxes = len(rois)
    if n_boxes == 0:
        return (
                features.new_zeros((0, features.shape[1])),
                features.new_zeros(0),)
    points = coordinates
    points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
    rois = rois.unsqueeze(0).expand(n_points, n_boxes, 7)
    face_distances = get_face_distances(points, rois)
    inside_condition = face_distances.min(dim=-1).values > 0
    min_pts_condition = inside_condition.sum(dim=0) >= min_pts_threshold
    inside_condition = inside_condition[:, min_pts_condition]
    rois = rois[0, min_pts_condition]
    nonzero = torch.nonzero(inside_condition)
    pooled_feat = torch_scatter.scatter_mean(
    features[nonzero[:, 0]],  
    nonzero[:, 1],          
    dim=0,                  
    )
    return pooled_feat, min_pts_condition

# Visualization functions
def write_obj(points, out_filename):
    """Write points into ``obj`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        if points.shape[1] == 6:
            c = points[i, 3:].astype(int)
            fout.write(
                'v %f %f %f %d %d %d\n' %
                (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

        else:
            fout.write('v %f %f %f\n' %
                    (points[i, 0], points[i, 1], points[i, 2]))
    fout.close()

def write_oriented_bbox(corners, labels, out_filename):
    """Export corners and labels to .obj file for meshlab.

    Args:
        corners(list[ndarray] or ndarray): [B x 8 x 3] corners of
            boxes for each scene
        labels(list[int]): labels of boxes for each scene
        out_filename(str): Filename.
    """
    colors = np.multiply([
        plt.cm.get_cmap('nipy_spectral', 19)((i * 5 + 11) % 18 + 1)[:3] for i in range(18)
    ], 255).astype(np.uint8).tolist()
    with open(out_filename, 'w') as file:
        for i, (corner, label) in enumerate(zip(corners, labels)):
            c = colors[label]
            for p in corner:
                file.write(f'v {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n')
            j = i * 8 + 1
            for k in [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                    [2, 3, 7, 6], [3, 0, 4, 7], [1, 2, 6, 5]]:
                file.write('f')
                for l in k:
                    file.write(f' {j + l}')
                file.write('\n')
    return

def write_oriented_bbox_v2(corners, labels, out_filename):
    """Export corners and labels to .obj file for meshlab.

    Args:
        corners(list[ndarray] or ndarray): [B x 8 x 3] corners of
            boxes for each scene
        labels(list[int]): labels of boxes for each scene
        out_filename(str): Filename.
    """
    colors = np.multiply([
        plt.cm.get_cmap('nipy_spectral', 19)((i * 5 + 11) % 18 + 1)[:3] for i in range(18)
    ], 255).astype(np.uint8).tolist()
    with open(out_filename, 'w') as file:
        for i, (corner, label) in enumerate(zip(corners, labels)):
            c = colors[label]
            for p in corner:
                file.write(f'v {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n')
            j = i * 8 + 1
            for k in [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                    [2, 3, 7, 6], [3, 0, 4, 7], [1, 2, 6, 5]]:
                file.write('f')
                for l in k:
                    file.write(f' {j + l}')
                file.write('\n')
    return



def show_result(points=None,
                gt_bboxes=None,
                gt_labels=None,
                pred_bboxes=None,
                pred_labels=None,
                fake_points = None,
                assign_points = None,
                out_dir=None,
                filename=None):
    """Convert results into format that is directly readable for meshlab.

    Args:
        points (np.ndarray): Points.
        gt_bboxes (np.ndarray): Ground truth boxes.
        pred_bboxes (np.ndarray): Predicted boxes.
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        show (bool): Visualize the results online. Defaults to False.
        snapshot (bool): Whether to save the online results. Defaults to False.
    """
    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    if points is not None:
        write_obj(points, osp.join(result_path, f'{filename}_points.obj'))
    
    if fake_points is not None:
        write_obj(fake_points, osp.join(result_path, f'{filename}_fake_points.obj'))

    if assign_points is not None:
        write_obj(assign_points, osp.join(result_path, f'{filename}_assign_points.obj'))

    if gt_bboxes is not None:
        write_oriented_bbox(gt_bboxes, gt_labels,
                            osp.join(result_path, f'{filename}_gt.obj'))

    if pred_bboxes is not None:
        write_oriented_bbox(pred_bboxes, pred_labels,
                            osp.join(result_path, f'{filename}_pred.obj'))
