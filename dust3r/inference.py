# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
import tqdm
import torch
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf
from typing import Dict, Any, Tuple, List


def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res


def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1,
                                     view2), _interleave_imgs(view2, view1))
    return view1, view2


def loss_of_one_batch(batch,
                      model,
                      criterion,
                      device,
                      symmetrize_batch=False,
                      use_amp=False,
                      ret=None):
    view1, view2 = batch
    ignore_keys = set([
        'depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'
    ])
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    if symmetrize_batch:
        view1, view2 = make_batch_symmetric(batch)

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        pred1, pred2 = model(view1, view2)

        # loss is supposed to be symmetric
        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(view1, view2, pred1,
                             pred2) if criterion is not None else None

    result = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2, loss=loss)
    return result[ret] if ret else result


@torch.no_grad()
def inference(pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
              model,
              device,
              batch_size=8,
              verbose=True) -> Dict[str, Any]:
    if verbose:
        print(f'>> Inference with model on {len(pairs)} image pairs')
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        # i in range(0, 2, 1) -> i: 0, 1
        res = loss_of_one_batch(collate_with_cat(pairs[i:i + batch_size]),
                                model, None, device)
        cpu_res = to_cpu(res)
        """ cpu_res: Dict[str, Any]
----- view1 -----
-- img --
img: torch.Size([1, 3, 288, 512])
-- true_shape --
true_shape: torch.Size([1, 2])
-- idx --
idx: [0]
-- instance --
instance: ['0']
----- view2 -----
-- img --
img: torch.Size([1, 3, 288, 512])
-- true_shape --
true_shape: torch.Size([1, 2])
-- idx --
idx: [1]
-- instance --
instance: ['1']
----- pred1 -----
-- pts3d --
pts3d: torch.Size([1, 288, 512, 3])
-- conf --
conf: torch.Size([1, 288, 512])
----- pred2 -----
-- conf --
conf: torch.Size([1, 288, 512])
-- pts3d_in_other_view --
pts3d_in_other_view: torch.Size([1, 288, 512, 3])
----- loss -----
loss: None
        """
        result.append(cpu_res)
    result = collate_with_cat(result, lists=multiple_shapes)
    return result
    """ output
    view1 (str): Dict
        img 
            tensor (2, 3, 288, 512)
                2: (1,0) 쌍에서 view1의 이미지 +  (0,1) 쌍에서 view1의 이미지
                288, 512: 이미지의 높이와 너비 ???
        true_shape
            tensor (2, 2)
                - 처음 2: (1,0) 쌍에서 view1의 이미지 shape 
                        - + (0,1) 쌍에서 view1의 이미지 shape
                - 두 번째 2: 이미지의 높이와 너비 = (288, 512)
        idx
            list: [1, 0]
                - 처음 쌍에서는 view1이 1 index 
                - 두번째 쌍에서는 view1이 0 index
        instance
            list: ['1', '0']
    view2 (str): Dict
        img 
            tensor (2, 3, 288, 512)
        true_shape
            tensor (2, 2)
        idx
            list: [0, 1]
        instance
            list: ['0', '1']
    pred1 (str): Dict
        pts3d
            tensor: (2, 288, 512, 3)
            - 2: (1,0) 쌍에서 view1의 pts (view1 좌표계 기준)
                +  (0,1) 쌍에서 view1의 pts (view 1 좌표계 기준)
        conf
            tensor: (2, 288, 512)
                - 2: (1,0) 쌍에서 view1의 pts의 신뢰도 값 (view1 좌표계 기준)
                    +  (0,1) 쌍에서 view1의 pts의 신뢰도 값 (view1 좌표계 기준)
    pred2 (str): Dict
        pts3d_in_other_view
            tensor: (2, 288, 512, 3)
                - 2: (1,0) 쌍에서 view2의 pts (view1 좌표계 기준)
                    +  (0,1) 쌍에서 view2의 pts (view1 좌표계 기준)
        conf
            tensor: (2, 288, 512)
                - 2: (1,0) 쌍에서 view2의 pts의 신뢰도 값 (view1 좌표계 기준)
                    +  (0,1) 쌍에서 view2의 pts의 신뢰도 값 (view1 좌표계 기준)
    loss (str): None

    """


def check_if_same_size(pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]):
    shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(
        shapes2[0] == s for s in shapes2)


def get_pred_pts3d(gt, pred, use_pose=False):
    if 'depth' in pred and 'pseudo_focal' in pred:
        try:
            pp = gt['camera_intrinsics'][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif 'pts3d' in pred:
        # pts3d from my camera
        pts3d = pred['pts3d']

    elif 'pts3d_in_other_view' in pred:
        # pts3d from the other camera, already transformed
        assert use_pose is True
        return pred['pts3d_in_other_view']  # return!

    if use_pose:
        camera_pose = pred.get('camera_pose')
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d


def find_opt_scaling(gt_pts1,
                     gt_pts2,
                     pr_pts1,
                     pr_pts2=None,
                     fit_mode='weiszfeld_stop_grad',
                     valid1=None,
                     valid2=None):
    assert gt_pts1.ndim == pr_pts1.ndim == 4
    assert gt_pts1.shape == pr_pts1.shape
    if gt_pts2 is not None:
        assert gt_pts2.ndim == pr_pts2.ndim == 4
        assert gt_pts2.shape == pr_pts2.shape

    # concat the pointcloud
    nan_gt_pts1 = invalid_to_nans(gt_pts1, valid1).flatten(1, 2)
    nan_gt_pts2 = invalid_to_nans(gt_pts2, valid2).flatten(
        1, 2) if gt_pts2 is not None else None

    pr_pts1 = invalid_to_nans(pr_pts1, valid1).flatten(1, 2)
    pr_pts2 = invalid_to_nans(pr_pts2, valid2).flatten(
        1, 2) if pr_pts2 is not None else None

    all_gt = torch.cat(
        (nan_gt_pts1,
         nan_gt_pts2), dim=1) if gt_pts2 is not None else nan_gt_pts1
    all_pr = torch.cat(
        (pr_pts1, pr_pts2), dim=1) if pr_pts2 is not None else pr_pts1

    dot_gt_pr = (all_pr * all_gt).sum(dim=-1)
    dot_gt_gt = all_gt.square().sum(dim=-1)

    if fit_mode.startswith('avg'):
        # scaling = (all_pr / all_gt).view(B, -1).mean(dim=1)
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
    elif fit_mode.startswith('median'):
        scaling = (dot_gt_pr / dot_gt_gt).nanmedian(dim=1).values
    elif fit_mode.startswith('weiszfeld'):
        # init scaling with l2 closed form
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (all_pr - scaling.view(-1, 1, 1) * all_gt).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip_(min=1e-8).reciprocal()
            # update the scaling with the new weights
            scaling = (w * dot_gt_pr).nanmean(dim=1) / (w * dot_gt_gt).nanmean(
                dim=1)
    else:
        raise ValueError(f'bad {fit_mode=}')

    if fit_mode.endswith('stop_grad'):
        scaling = scaling.detach()

    scaling = scaling.clip(min=1e-3)
    # assert scaling.isfinite().all(), bb()
    return scaling
