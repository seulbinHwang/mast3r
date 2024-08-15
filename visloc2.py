#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Simple visloc script
# --------------------------------------------------------
import numpy as np
import random
import argparse
from tqdm import tqdm
import math
from typing import List, Dict, Any

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid, geotrf

from dust3r_visloc.datasets import *
from dust3r_visloc.localization import run_pnp
from dust3r_visloc.evaluation import get_pose_error, aggregate_stats, export_results

"""
위 코드는 Visual Localization (visloc) 작업을 수행하는 스크립트로, 
    특정 데이터셋에서 이미지의 위치를 추정하고 그 성능을 평가

### 코드의 목적
1. **Visual Localization 성능 평가**:
   - DUSt3R 모델을 사용하여 이미지의 위치를 추정하고, 그 성능을 평가

2. **PnP 문제 해결**:
   - "2D-3D 점 매칭을 기반으로 카메라 포즈를 추정하는 PnP 알고리즘을 사용"

3. **결과 시각화 및 저장**:
   - 매칭 결과를 시각화하고, 평가 결과를 파일로 저장


### 주요 라이브러리 및 모듈
10. **dust3r_visloc.localization**: 
    PnP 문제를 해결하는 함수가 포함
11. **dust3r_visloc.evaluation**: 
    위치 추정 오류를 계산하고 결과를 집계하는 함수가 포함

### 주요 함수 및 변수

2. **main 함수**:
   - 명령줄 인자를 파싱하여 설정을 로드
   - 모델을 로드하고, 데이터셋을 설정
   - "각 이미지 쌍"에 대해 다음 작업을 수행
     - 이미지 데이터를 준비하고, 모델을 사용하여 추론을 수행
     - 추론 결과에서 2D-3D 매칭 점을 추출
     - PnP 알고리즘을 사용하여 카메라 포즈를 추정
     - 위치 추정 오류를 계산
     - 결과를 저장하고, 통계를 집계

3. **추론 및 매칭**:
   - `inference` 함수를 사용하여 이미지 쌍에 대한 특징 추출 및 매칭을 수행
   - 추론 결과에서 신뢰도가 높은 매칭 점을 필터링하고, 2D-2D 매칭을 수행
   - 매칭 결과를 시각화할 수 있습니다 (디버깅 목적).

4. **PnP 알고리즘**:
   - `run_pnp` 함수를 사용하여 2D-3D 점 매칭을 기반으로 카메라 포즈를 추정
   - OpenCV, poselib, pycolmap 세 가지 모드를 지원
   - PnP 문제를 해결하고, 추정된 카메라 포즈를 반환

5. **결과 집계 및 저장**:
   - 각 이미지 쌍에 대해 위치 추정 오류(절대 변위 오류 및 절대 각도 오류)를 계산
   - `aggregate_stats` 함수를 사용하여 전체 데이터셋에 대한 성능 통계를 집계
   - `export_results` 함수를 사용하여 결과를 파일로 저장

# 7-scenes:
# scene in 'chess' 'fire' 'heads' 'office' 'pumpkin' 'redkitchen' 'stairs'
python3 visloc.py 
    --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt 
    --dataset "VislocSevenScenes('/path/to/prepared/7-scenes/', subscene='${scene}', pairsfile='APGeM-LM18_top20', topk=1)" 
    --pnp_mode poselib 
    --reprojection_error_diag_ratio 0.008 
    --output_dir /path/to/output/7-scenes/${scene}/loc

"""
def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="visloc dataset to eval")
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights",
                                type=str,
                                help="path to the model weights",
                                default=None)
    parser_weights.add_argument("--model_name",
                                type=str,
                                help="name of the model weights",
                                choices=[
                                    "DUSt3R_ViTLarge_BaseDecoder_512_dpt",
                                    "DUSt3R_ViTLarge_BaseDecoder_512_linear",
                                    "DUSt3R_ViTLarge_BaseDecoder_224_linear"
                                ])
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=3.0,
        help="confidence values higher than threshold are invalid")
    parser.add_argument("--device",
                        type=str,
                        default='cuda',
                        help="pytorch device")
    parser.add_argument("--pnp_mode",
                        type=str,
                        default="cv2",
                        choices=['cv2', 'poselib', 'pycolmap'],
                        help="pnp lib to use")
    parser_reproj = parser.add_mutually_exclusive_group()
    parser_reproj.add_argument("--reprojection_error",
                               type=float,
                               default=5.0,
                               help="pnp reprojection error")
    parser_reproj.add_argument(
        "--reprojection_error_diag_ratio",
        type=float,
        default=None,
        help="pnp reprojection error as a ratio of the diagonal of the image")

    parser.add_argument("--pnp_max_points",
                        type=int,
                        default=100_000,
                        help="pnp maximum number of points kept")
    parser.add_argument("--viz_matches",
                        type=int,
                        default=0,
                        help="debug matches")

    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="output path")
    parser.add_argument("--output_label",
                        type=str,
                        default='',
                        help="prefix for results files")
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    conf_thr = args.confidence_threshold
    device = args.device
    pnp_mode = args.pnp_mode
    reprojection_error = args.reprojection_error
    reprojection_error_diag_ratio = args.reprojection_error_diag_ratio
    pnp_max_points = args.pnp_max_points
    viz_matches = args.viz_matches

    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(
        args.device)

    dataset = eval(args.dataset)
    dataset.set_resolution(model)

    query_names = []
    poses_pred = []
    for idx in tqdm(range(len(dataset))):
        views = dataset[(idx)]  # 0 is the query
        query_view: Dict[str, Any] = views[0]
        map_views: Dict[str, Any] = views[1:]
        query_names.append(query_view['image_name'])

        query_pts2d = []
        query_pts3d = []
        for map_view in map_views:
            # prepare batch
            imgs = []
            for idx, img in enumerate(
                [query_view['rgb_rescaled'], map_view['rgb_rescaled']]):
                imgs.append(
                    dict(img=img.unsqueeze(0),
                         true_shape=np.int32([img.shape[1:]]),
                         idx=idx,
                         instance=str(idx)))
            """
   - `inference` 함수를 사용하여 이미지 쌍에 대한 특징 추출 및 매칭을 수행
   - 추론 결과에서 신뢰도가 높은 매칭 점을 필터링하고, 2D-2D 매칭을 수행
   - 매칭 결과를 시각화할 수 있습니다 (디버깅 목적).
            """
            output = inference([tuple(imgs)],
                               model,
                               device,
                               batch_size=1,
                               verbose=False)
            pred1, pred2 = output['pred1'], output['pred2']
            confidence_masks = [
                pred1['conf'].squeeze(0) >= conf_thr,
                (pred2['conf'].squeeze(0) >= conf_thr) &
                map_view['valid_rescaled']
            ]
            pts3d = [
                pred1['pts3d'].squeeze(0),
                pred2['pts3d_in_other_view'].squeeze(0)
            ]

            # find 2D-2D matches between the two images
            pts2d_list, pts3d_list = [], []
            for i in range(2):
                conf_i = confidence_masks[i].cpu().numpy()
                true_shape_i = imgs[i]['true_shape'][0]
                pts2d_list.append(
                    xy_grid(true_shape_i[1], true_shape_i[0])[conf_i])
                pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])

            PQ, PM = pts3d_list[0], pts3d_list[1]
            if len(PQ) == 0 or len(PM) == 0:
                continue
            reciprocal_in_PM, nnM_in_PQ, num_matches = find_reciprocal_matches(
                PQ, PM)
            if viz_matches > 0:
                print(f'found {num_matches} matches')
            matches_im1 = pts2d_list[1][reciprocal_in_PM]
            matches_im0 = pts2d_list[0][nnM_in_PQ][reciprocal_in_PM]
            valid_pts3d = map_view['pts3d_rescaled'][matches_im1[:, 1],
                                                     matches_im1[:, 0]]

            # from cv2 to colmap
            matches_im0 = matches_im0.astype(np.float64)
            matches_im1 = matches_im1.astype(np.float64)
            matches_im0[:, 0] += 0.5
            matches_im0[:, 1] += 0.5
            matches_im1[:, 0] += 0.5
            matches_im1[:, 1] += 0.5
            # rescale coordinates
            matches_im0 = geotrf(query_view['to_orig'], matches_im0, norm=True)
            matches_im1 = geotrf(query_view['to_orig'], matches_im1, norm=True)
            # from colmap back to cv2
            matches_im0[:, 0] -= 0.5
            matches_im0[:, 1] -= 0.5
            matches_im1[:, 0] -= 0.5
            matches_im1[:, 1] -= 0.5

            # visualize a few matches
            if viz_matches > 0:
                viz_imgs = [
                    np.array(query_view['rgb']),
                    np.array(map_view['rgb'])
                ]
                from matplotlib import pyplot as pl
                n_viz = viz_matches
                match_idx_to_viz = np.round(
                    np.linspace(0, num_matches - 1, n_viz)).astype(int)
                viz_matches_im0, viz_matches_im1 = matches_im0[
                    match_idx_to_viz], matches_im1[match_idx_to_viz]

                H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
                img0 = np.pad(viz_imgs[0],
                              ((0, max(H1 - H0, 0)), (0, 0), (0, 0)),
                              'constant',
                              constant_values=0)
                img1 = np.pad(viz_imgs[1],
                              ((0, max(H0 - H1, 0)), (0, 0), (0, 0)),
                              'constant',
                              constant_values=0)
                img = np.concatenate((img0, img1), axis=1)
                pl.figure()
                pl.imshow(img)
                cmap = pl.get_cmap('jet')
                for i in range(n_viz):
                    (x0, y0), (x1,
                               y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
                    pl.plot([x0, x1 + W0], [y0, y1],
                            '-+',
                            color=cmap(i / (n_viz - 1)),
                            scalex=False,
                            scaley=False)
                pl.show(block=True)

            if len(valid_pts3d) == 0:
                pass
            else:
                query_pts3d.append(valid_pts3d.cpu().numpy())
                query_pts2d.append(matches_im0)

        if len(query_pts2d) == 0:
            success = False
            pr_querycam_to_world = None
        else:
            query_pts2d = np.concatenate(query_pts2d, axis=0).astype(np.float32)
            query_pts3d = np.concatenate(query_pts3d, axis=0)
            if len(query_pts2d) > pnp_max_points:
                idxs = random.sample(range(len(query_pts2d)), pnp_max_points)
                query_pts3d = query_pts3d[idxs]
                query_pts2d = query_pts2d[idxs]

            W, H = query_view['rgb'].size
            if reprojection_error_diag_ratio is not None:
                reprojection_error_img = reprojection_error_diag_ratio * math.sqrt(
                    W**2 + H**2)
            else:
                reprojection_error_img = reprojection_error
            success, pr_querycam_to_world = run_pnp(query_pts2d,
                                                    query_pts3d,
                                                    query_view['intrinsics'],
                                                    query_view['distortion'],
                                                    pnp_mode,
                                                    reprojection_error_img,
                                                    img_size=[W, H])
        poses_pred.append(pr_querycam_to_world)
    print("poses_pred: ", poses_pred)
