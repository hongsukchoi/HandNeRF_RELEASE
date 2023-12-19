from typing import Optional
import torch
import numpy as np
import lpips
from torch.nn import functional as F
from torchmetrics.functional import structural_similarity_index_measure

from config import cfg
from constant import CLIP_NORMALIZE, CLIP_SIZE, OBJ_SEG_IDX, HAND_SEG_IDX, OBJ_SEMANTIC_IDX, HAND_SEMANTIC_IDX

if cfg.is_test:
    if cfg.test_mode == 'render_dataset':
        # comment this during training if memoery insufficient
        lpips_nets = {
            # 'alex': lpips.LPIPS(net='alex').cuda(),  # best forward scores,
            'vgg': lpips.LPIPS(net='vgg').cuda()
        }
    else:
        pass

# evaluation for segmentation; compute iou from binary masks
def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    preds: list (num_classes; H, W) np.ndarray
    labels: list (num_classes; H, W) np.ndarray
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    
    if len(ious) == 0:
        raise ValueError("Not enough pairs to compare")

    iou = np.mean(ious)  # mean accross images if per_image

    return 100 * iou

# from IHOI
def calc_3d_metrics(cd_function: object, pred_pts: torch.Tensor, gt_pts: torch.Tensor, num_samples=10000, th=None):
    """
    Calculate bathced F-score between  2 point clouds
    https://github.com/lmb-freiburg/what3d/blob/1f8e6bf5b1334166b02d9b86b14354edb77992bd/util.py#L54

    pred_pts: (batch_size, N, 3)
    gt_pts: (batch_size, N, 3)
    """
    if th is None:
        th_list = [.2/100, .5 / 100, 1./100, 1./50, 1./20, 1./10]
    else:
        th_list = th

    d1, d2 = cd_function(pred_pts, gt_pts, bidirectional=True, batch_reduction=None, point_reduction=None)

    d1 = torch.sqrt(d1)
    d2 = torch.sqrt(d2)
    res_list = []
    for th in th_list:
        if d1.size(1) and d2.size(1):
            recall = torch.sum(d2 < th, dim=-1).to(gt_pts) / num_samples  # recall knn(gt, pred) gt->pred
            # precision knn(pred, gt) pred-->
            precision = torch.sum(d1 < th, dim=-1).to(gt_pts) / num_samples

            eps = 1e-6
            fscore = 2 * recall * precision / (recall + precision + eps)
            # res_list.append([fscore, precision, recall])
            res_list.append(fscore.tolist())
        else:
            raise ValueError("d1 and d2 should be in equal length but got %d %d" % (
                d1.size(1), d2.size(1)))
    d = ((d1 ** 2).mean(1) + (d2 ** 2).mean(1)).tolist()
    return res_list, d

# from Pytorch3d
def calc_bce(
    pred: torch.Tensor,
    gt: torch.Tensor,
    equal_w: bool = True,
    pred_eps: float = 0.01,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculates the binary cross entropy.
    """
    if pred_eps > 0.0:
        # up/low bound the predictions
        pred = torch.clamp(pred, pred_eps, 1.0 - pred_eps)

    if mask is None:
        mask = torch.ones_like(gt)

    if equal_w:
        mask_fg = (gt > 0.5).float() * mask
        mask_bg = (1 - mask_fg) * mask
        weight = mask_fg / mask_fg.sum().clamp(1.0) + mask_bg / \
            mask_bg.sum().clamp(1.0)
        # weight sum should be at this point ~2
        # pyre-fixme[58]: `/` is not supported for operand types `int` and `Tensor`.
        weight = weight * (weight.numel() / weight.sum().clamp(1.0))
    else:
        weight = torch.ones_like(gt) * mask
    
    return F.binary_cross_entropy(pred, gt, reduction="mean", weight=weight)

# custom
# regularize the background depth, which is the masked area
# make to 0. purpose is to make zero opacity of the background ray
def calc_depth_reg_loss(depth, fg_mask):
    """
    depth: (batch_size, num_rays, 1)
    fg_mask: (batch_size, num_rays)
    
    """
    bg_mask = ~fg_mask.bool()
    bg_depth = depth[bg_mask]  # (num_bg_rays, 1)

    # depth_reg_loss = (max_depth - bg_depth).mean()
    depth_reg_loss = bg_depth.abs().mean()  # abs() clip to zero; actually don't need this
    
    return depth_reg_loss


# custom
# regularize the background ray density, which is the masked area
# make to 0. purpose is to make zero opacity of the background ray
def calc_opacity_reg_loss(density, fg_mask):
    """
    density: (batch_size, num_rays, cfg.N_samples)
    fg_mask: (batch_size, num_rays)
    
    """

    bg_mask = ~fg_mask.bool()
    bg_density = density[bg_mask]  # (num_bg_rays, -1)

    density_reg_loss = bg_density.abs().sum(dim=-1).mean()  # abs() clip to zero; actually don't need this

    return density_reg_loss



# custom
def calc_semantic_ce(
    pred: torch.Tensor,
    gt: torch.Tensor,
    weight: Optional[torch.Tensor] = None
):
    """
    pred: (batch_size, num_rays, 3)
    gt: (batch_size, num_rays), raw sematic mask from dataset annotaiton. needs parsing
    weight: (3,), background, object, hand 
    """

    gt[gt == OBJ_SEG_IDX] = OBJ_SEMANTIC_IDX
    # TEMP
    # gt[gt == OBJ_SEG_IDX] = 0
    gt[gt == HAND_SEG_IDX] = HAND_SEMANTIC_IDX
    # background is already zero

    return F.cross_entropy(pred.view(-1,3), gt.view(-1), weight=weight, reduction="mean")

# from Pytorch3d
def calc_mse(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculates the mean square error between tensors `x` and `y`.
    If mask not None and use_mask False, it means using mask as dummy (replacement of corrupted data, see ray_sampling.py) classifier 
    """
    if mask is None:
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        return torch.mean((x - y) ** 2)
    else:
        return (((x - y) ** 2) * mask).sum() / mask.expand_as(x).sum().clamp(1e-5)


def calc_psnr(mse: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`.
    """
    psnr = torch.log10(mse.clamp(1e-10)) * (-10.0)
    return psnr


def calc_clip_loss(clip_model: object, rendered_raw_img: torch.Tensor, mask_at_box: torch.Tensor, clip_target_img: torch.Tensor) -> torch.Tensor:
    """
    clip: clip model
    rendered_raw_img: (1, H*W-?, 3), normalized to 0~1
    mask_at_box: (1, H*W)
    clip_target_img: an object image without hand, (1, 224, 224 3)

    // Returns //
    cosine simularity loss
    """

    # reshape and resize
    rendered_img = torch.zeros(*mask_at_box.shape, 3).to(rendered_raw_img)
    rendered_img[mask_at_box] = rendered_raw_img
    rendered_img = rendered_img.reshape(1, cfg.clip_render_img_shape[1], cfg.clip_render_img_shape[0], 3).permute(0, 3, 1, 2)  # (1,3, H, W)

    rendered_resized_img = F.interpolate(rendered_img, size=(CLIP_SIZE, CLIP_SIZE), mode='bicubic')
    target_img = clip_target_img.permute(0, 3, 1, 2)  # (1, 3, 224, 224)

    # normalize 
    norm_rendered_img = CLIP_NORMALIZE(rendered_resized_img)  
    norm_target_img = CLIP_NORMALIZE(target_img)

    # check text - image matching
    # text_inputs = clip.tokenize(f"a photo of a masterchef coffee can").to(norm_rendered_img).int()

    # encode; (1,512)
    with torch.no_grad():
        target_emb = clip_model.encode_image(norm_target_img)
        # text_emb = clip_model.encode_text(text_inputs)
        
    rendered_emb = clip_model.encode_image(norm_rendered_img)
    # the more similar, the better
    consistency_loss = -torch.cosine_similarity(target_emb, rendered_emb, dim=-1)

    return consistency_loss 

# the lower the more similar
def calc_lpips(img1: torch.Tensor, img2: torch.Tensor, option: str = 'vgg'):
    """
    image should be RGB, IMPORTANT: normalized to [-1,1]

    img1: (1, 3, H, W)
    img2: (1, 3, H, W)
    """

    net = lpips_nets[option]
    lpips_score = net(img1, img2)

    return lpips_score


# the higher the more similar
# cacluate the ssim, use the default kernel size 11
# data_range: -1 ~ 1
def calc_ssim(img1: torch.Tensor, img2: torch.Tensor, data_range: Optional[float] = 2):
    """
    image should be RGB, IMPORTANT: normalized to [-1,1]

    img1: (N, 3, H, W)
    img2: (N, 3, H, W)
    """

    ssim = structural_similarity_index_measure(img1, img2, data_range=data_range)  # default reduction='elementwise_mean',  other options: 'sum', 'none'
    return ssim