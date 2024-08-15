# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Main class for the implementation of the global alignment
# --------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn

from dust3r.cloud_opt.base_opt import BasePCOptimizer
from dust3r.utils.geometry import xy_grid, geotrf
from dust3r.utils.device import to_cpu, to_numpy
"""
### `PointCloudOptimizer` 클래스의 목적과 내용

#### 목적

`PointCloudOptimizer` 클래스는 여러 이미지 쌍으로부터 관찰된 포인트 클라우드 데이터를 사용하여 
    전체 장면을 최적화하는 도구
    이 클래스는 이미지 간의 관계를 그래프 구조로 표현하고, 
    각 이미지에 대한 카메라 포즈와 깊이 맵을 최적화하여 전체 장면의 일관된 3D 재구성을 제공

#### 요약

- **전체 장면 최적화**: 여러 이미지 쌍을 사용하여 전체 3D 장면을 최적화
- **카메라 포즈 및 깊이 맵 최적화**: 
    각 이미지의 카메라 포즈와 깊이 맵을 최적화하여 일관된 3D 재구성을 제공
- **초점 거리 및 주점 설정**: 
    각 이미지의 카메라 내적 파라미터를 설정하고 최적화
- **손실 계산**: 
    이미지 쌍 간의 예측된 3D 포인트 클라우드를 정렬하고, 최적화 손실을 계산

#### 주요 기능 및 내용

1. **초기화 (`__init__` 메서드)**:
   - 클래스 초기화 시, 카메라 포즈, 초점 거리, 주점(principal point) 등을 매개변수로 받아 설정
   - 각 이미지의 깊이 맵, 포즈, 초점 거리 등을 최적화할 파라미터로 설정
   - 각 이미지의 해상도와 크기를 기반으로 최대 영역을 계산하고, 이를 기준으로 파라미터 스택을 생성

2. **preset_pose 메서드**:
   - 주어진 카메라 포즈를 설정합니다.
   - 카메라 포즈를 설정할 때, 주어진 마스크에 따라 선택된 포즈만 설정

3. **preset_focal 메서드**:
   - 주어진 초점 거리를 설정
   - 초점 거리를 설정할 때, 주어진 마스크에 따라 선택된 초점 거리만 설정

4. **preset_principal_point 메서드**:
   - 주어진 주점을 설정합니다.
   - 주점을 설정할 때, 주어진 마스크에 따라 선택된 주점만 설정합니다.

5. **get_focals 메서드**:
   - 현재 설정된 모든 초점 거리를 반환합니다.

6. **get_known_focal_mask 메서드**:
   - 초점 거리가 고정된(fixed) 여부를 나타내는 마스크를 반환합니다.

7. **get_principal_points 메서드**:
   - 현재 설정된 모든 주점을 반환합니다.

8. **get_intrinsics 메서드**:
   - 카메라 내적 행렬(intrinsics)을 반환합니다.

9. **get_im_poses 메서드**:
   - 각 이미지의 카메라 포즈(월드 좌표계에서 카메라 좌표계로의 변환)를 반환합니다.

10. **_set_depthmap 메서드**:
    - 주어진 깊이 맵을 설정합니다.

11. **get_depthmaps 메서드**:
    - 모든 이미지의 깊이 맵을 반환합니다.

12. **depth_to_pts3d 메서드**:
    - 깊이 맵을 3D 포인트 클라우드로 변환하여 반환합니다.

13. **get_pts3d 메서드**:
    - 3D 포인트 클라우드를 반환합니다.

14. **forward 메서드**:
    - 모델의 순전파(forward) 연산을 수행합니다.
    - 이미지 쌍 간의 예측된 포인트 클라우드를 회전 및 변환하여 정렬하고, 최적화 손실을 계산합니다.

"""


class PointCloudOptimizer(BasePCOptimizer):
    """ Optimize a global scene, given a list of pairwise observations.
    Graph node: images
    Graph edges: observations = (pred1, pred2)
    """

    def __init__(self, *args, optimize_pp=False, focal_break=20, **kwargs):
        super().__init__(*args, **kwargs)

        self.has_im_poses = True  # by definition of this class
        self.focal_break = focal_break

        # adding thing to optimize
        self.im_depthmaps = nn.ParameterList(
            torch.randn(H, W) / 10 - 3 for H, W in self.imshapes)  # log(depth)
        self.im_poses = nn.ParameterList(
            self.rand_pose(self.POSE_DIM)
            for _ in range(self.n_imgs))  # camera poses
        self.im_focals = nn.ParameterList(
            torch.FloatTensor([self.focal_break * np.log(max(H, W))])
            for H, W in self.imshapes)  # camera intrinsics
        self.im_pp = nn.ParameterList(
            torch.zeros((2,)) for _ in range(self.n_imgs))  # camera intrinsics
        self.im_pp.requires_grad_(optimize_pp)

        self.imshape = self.imshapes[0]
        im_areas = [h * w for h, w in self.imshapes]
        self.max_area = max(im_areas)

        # adding thing to optimize
        self.im_depthmaps = ParameterStack(self.im_depthmaps,
                                           is_param=True,
                                           fill=self.max_area)
        self.im_poses = ParameterStack(self.im_poses, is_param=True)
        self.im_focals = ParameterStack(self.im_focals, is_param=True)
        self.im_pp = ParameterStack(self.im_pp, is_param=True)
        self.register_buffer(
            '_pp', torch.tensor([(w / 2, h / 2) for h, w in self.imshapes]))
        self.register_buffer(
            '_grid',
            ParameterStack(
                [xy_grid(W, H, device=self.device) for H, W in self.imshapes],
                fill=self.max_area))

        # pre-compute pixel weights
        self.register_buffer(
            '_weight_i',
            ParameterStack(
                [self.conf_trf(self.conf_i[i_j]) for i_j in self.str_edges],
                fill=self.max_area))
        self.register_buffer(
            '_weight_j',
            ParameterStack(
                [self.conf_trf(self.conf_j[i_j]) for i_j in self.str_edges],
                fill=self.max_area))

        # precompute aa
        self.register_buffer(
            '_stacked_pred_i',
            ParameterStack(self.pred_i, self.str_edges, fill=self.max_area))
        self.register_buffer(
            '_stacked_pred_j',
            ParameterStack(self.pred_j, self.str_edges, fill=self.max_area))
        self.register_buffer('_ei', torch.tensor([i for i, j in self.edges]))
        self.register_buffer('_ej', torch.tensor([j for i, j in self.edges]))
        self.total_area_i = sum([im_areas[i] for i, j in self.edges])
        self.total_area_j = sum([im_areas[j] for i, j in self.edges])

    def _check_all_imgs_are_selected(self, msk):
        assert np.all(self._get_msk_indices(msk) == np.arange(
            self.n_imgs)), 'incomplete mask!'

    def preset_pose(self, known_poses, pose_msk=None):  # cam-to-world
        self._check_all_imgs_are_selected(pose_msk)

        if isinstance(known_poses, torch.Tensor) and known_poses.ndim == 2:
            known_poses = [known_poses]
        for idx, pose in zip(self._get_msk_indices(pose_msk), known_poses):
            if self.verbose:
                print(f' (setting pose #{idx} = {pose[:3,3]})')
            self._no_grad(self._set_pose(self.im_poses, idx,
                                         torch.tensor(pose)))

        # normalize scale if there's less than 1 known pose
        n_known_poses = sum((p.requires_grad is False) for p in self.im_poses)
        self.norm_pw_scale = (n_known_poses <= 1)

        self.im_poses.requires_grad_(False)
        self.norm_pw_scale = False

    def preset_focal(self, known_focals, msk=None):
        self._check_all_imgs_are_selected(msk)

        for idx, focal in zip(self._get_msk_indices(msk), known_focals):
            if self.verbose:
                print(f' (setting focal #{idx} = {focal})')
            self._no_grad(self._set_focal(idx, focal))

        self.im_focals.requires_grad_(False)

    def preset_principal_point(self, known_pp, msk=None):
        self._check_all_imgs_are_selected(msk)

        for idx, pp in zip(self._get_msk_indices(msk), known_pp):
            if self.verbose:
                print(f' (setting principal point #{idx} = {pp})')
            self._no_grad(self._set_principal_point(idx, pp))

        self.im_pp.requires_grad_(False)

    def _get_msk_indices(self, msk):
        if msk is None:
            return range(self.n_imgs)
        elif isinstance(msk, int):
            return [msk]
        elif isinstance(msk, (tuple, list)):
            return self._get_msk_indices(np.array(msk))
        elif msk.dtype in (bool, torch.bool, np.bool_):
            assert len(msk) == self.n_imgs
            return np.where(msk)[0]
        elif np.issubdtype(msk.dtype, np.integer):
            return msk
        else:
            raise ValueError(f'bad {msk=}')

    def _no_grad(self, tensor):
        assert tensor.requires_grad, 'it must be True at this point, otherwise no modification occurs'

    def _set_focal(self, idx, focal, force=False):
        param = self.im_focals[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = self.focal_break * np.log(focal)
        return param

    def get_focals(self):
        log_focals = torch.stack(list(self.im_focals), dim=0)
        return (log_focals / self.focal_break).exp()

    def get_known_focal_mask(self):
        return torch.tensor([not (p.requires_grad) for p in self.im_focals])

    def _set_principal_point(self, idx, pp, force=False):
        param = self.im_pp[idx]
        H, W = self.imshapes[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = to_cpu(to_numpy(pp) - (W / 2, H / 2)) / 10
        return param

    def get_principal_points(self):
        return self._pp + 10 * self.im_pp

    def get_intrinsics(self):
        K = torch.zeros((self.n_imgs, 3, 3), device=self.device)
        focals = self.get_focals().flatten()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, :2, 2] = self.get_principal_points()
        K[:, 2, 2] = 1
        return K

    def get_im_poses(self):  # cam to world
        cam2world = self._get_poses(self.im_poses)
        return cam2world

    def _set_depthmap(self, idx, depth, force=False):
        depth = _ravel_hw(depth, self.max_area)

        param = self.im_depthmaps[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = depth.log().nan_to_num(neginf=0)
        return param

    def get_depthmaps(self, raw=False):
        res = self.im_depthmaps.exp()
        if not raw:
            res = [
                dm[:h * w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)
            ]
        return res

    def depth_to_pts3d(self):
        # Get depths and  projection params if not provided
        focals = self.get_focals()
        pp = self.get_principal_points()
        im_poses = self.get_im_poses()
        depth = self.get_depthmaps(raw=True)

        # get pointmaps in camera frame
        rel_ptmaps = _fast_depthmap_to_pts3d(depth, self._grid, focals, pp=pp)
        # project to world frame
        return geotrf(im_poses, rel_ptmaps)

    def get_pts3d(self, raw=False):
        res = self.depth_to_pts3d()
        if not raw:
            res = [
                dm[:h * w].view(h, w, 3)
                for dm, (h, w) in zip(res, self.imshapes)
            ]
        return res

    def forward(self):
        pw_poses = self.get_pw_poses()  # cam-to-world
        pw_adapt = self.get_adaptors().unsqueeze(1)
        proj_pts3d = self.get_pts3d(raw=True)

        # rotate pairwise prediction according to pw_poses
        aligned_pred_i = geotrf(pw_poses, pw_adapt * self._stacked_pred_i)
        aligned_pred_j = geotrf(pw_poses, pw_adapt * self._stacked_pred_j)

        # compute the less
        li = self.dist(proj_pts3d[self._ei],
                       aligned_pred_i,
                       weight=self._weight_i).sum() / self.total_area_i
        lj = self.dist(proj_pts3d[self._ej],
                       aligned_pred_j,
                       weight=self._weight_j).sum() / self.total_area_j

        return li + lj


def _fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)
    assert focal.shape == (len(depth), 1, 1)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == depth.shape + (2,)
    depth = depth.unsqueeze(-1)
    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)


def ParameterStack(params, keys=None, is_param=None, fill=0):
    if keys is not None:
        params = [params[k] for k in keys]

    if fill > 0:
        params = [_ravel_hw(p, fill) for p in params]

    requires_grad = params[0].requires_grad
    assert all(p.requires_grad == requires_grad for p in params)

    params = torch.stack(list(params)).float().detach()
    if is_param or requires_grad:
        params = nn.Parameter(params)
        params.requires_grad_(requires_grad)
    return params


def _ravel_hw(tensor, fill=0):
    # ravel H,W
    tensor = tensor.view((tensor.shape[0] * tensor.shape[1],) +
                         tensor.shape[2:])

    if len(tensor) < fill:
        tensor = torch.cat(
            (tensor,
             tensor.new_zeros((fill - len(tensor),) + tensor.shape[1:])))
    return tensor


def acceptable_focal_range(H, W, minf=0.5, maxf=3.5):
    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2)
                             )  # size / 1.1547005383792515
    return minf * focal_base, maxf * focal_base


def apply_mask(img, msk):
    img = img.copy()
    img[msk] = 0
    return img
