# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# main pnp code
# --------------------------------------------------------
import numpy as np
import quaternion
import cv2
from packaging import version

from dust3r.utils.geometry import opencv_to_colmap_intrinsics
"""
이 코드는 이미지에서 추출한 2D 점과 3D 점을 기반으로, 
"카메라의 포즈"를 추정하는 PnP(포인트 투 포인트) 문제를 해결하는 함수인 `run_pnp`를 구현
이 함수는 세 가지 모드를 지원합니다: 
    `cv2`, `poselib`, `pycolmap`. 
각각의 모드는 다른 라이브러리를 사용하여 PnP 문제를 해결

### 주요 라이브러리 및 모듈
3. **cv2 (OpenCV)**: PnP 문제를 해결하는 기능을 제공
5. **dust3r.utils.geometry**: 
    OpenCV 카메라 내재 파라미터를 COLMAP 형식으로 변환하는 함수가 포함되어 있음
6. **poselib**: 
    포즈 추정 라이브러리로, PnP 문제를 해결하기 위한 기능을 제공
7. **pycolmap**: 
    COLMAP을 Python에서 사용할 수 있게 하는 라이브러리로, 이미지의 포즈 추정 기능을 제공

### 함수 `run_pnp`의 주요 기능

1. **입력 인자**:
   - `pts2D`: 이미지에서 추출한 2D 점 좌표.
   - `pts3D`: 해당 2D 점과 대응되는 3D 점 좌표.
   - `K`: 카메라 내재 파라미터 행렬.
   - `distortion`: 카메라의 왜곡 계수.
   - `mode`: 
        - 사용할 PnP 솔버의 모드 (`cv2`, `poselib`, `pycolmap`).
   - `reprojectionError`: 
        - RANSAC 알고리즘에서 허용할 재투영 오차.
   - `img_size`: 이미지의 크기.

2. **모드별 PnP 해결**:
   - `cv2` 모드:
     - OpenCV의 `solvePnPRansac` 함수를 사용하여 3D-2D 점 매칭을 기반으로 카메라 포즈를 추정
     - RANSAC 알고리즘을 사용하여 외부 파라미터(회전 및 변환 벡터)를 추정
     - 왜곡 계수가 주어지면 `undistortPoints` 함수를 사용하여 2D 점을 보정
     - 성공 여부와 함께 "카메라 좌표계"를 "월드 좌표계로 변환"하는 행렬을 반환

   - `poselib` 모드:
     - `poselib` 라이브러리를 사용하여 절대 포즈를 추정
     - COLMAP 형식의 "카메라 내재 파라미터"로 변환한 후, 
        - 왜곡 계수가 있으면 이를 포함하여 카메라 모델을 정의
     - `poselib.estimate_absolute_pose` 함수를 사용하여 포즈를 추정하고, 
            성공 여부와 함께 카메라 좌표계를 월드 좌표계로 변환하는 행렬을 반환

   - `pycolmap` 모드:
     - `pycolmap` 라이브러리를 사용하여 절대 포즈를 추정
     - COLMAP 형식의 카메라 내재 파라미터로 변환하고, 카메라 모델을 정의
     - `pycolmap.absolute_pose_estimation` 함수를 사용하여 포즈를 추정하고, 
        성공 여부와 함께 카메라 좌표계를 월드 좌표계로 변환하는 행렬을 반환
     - 추정된 회전 행렬을 쿼터니언 형식으로 변환하여 결과를 반환
"""
try:
    import poselib  # noqa
    HAS_POSELIB = True
except Exception as e:
    HAS_POSELIB = False

try:
    import pycolmap  # noqa
    version_number = pycolmap.__version__
    if version.parse(version_number) < version.parse("0.5.0"):
        HAS_PYCOLMAP = False
    else:
        HAS_PYCOLMAP = True
except Exception as e:
    HAS_PYCOLMAP = False


def run_pnp(pts2D,
            pts3D,
            K,
            distortion=None,
            mode='cv2',
            reprojectionError=5,
            img_size=None):
    """
    use OPENCV model for distortion (4 values)
    """
    assert mode in ['cv2', 'poselib', 'pycolmap']
    try:
        if len(pts2D) > 4 and mode == "cv2":
            confidence = 0.9999
            iterationsCount = 10_000
            if distortion is not None:
                cv2_pts2ds = np.copy(pts2D)
                cv2_pts2ds = cv2.undistortPoints(cv2_pts2ds,
                                                 K,
                                                 np.array(distortion),
                                                 R=None,
                                                 P=K)
                pts2D = cv2_pts2ds.reshape((-1, 2))

            success, r_pose, t_pose, _ = cv2.solvePnPRansac(
                pts3D,
                pts2D,
                K,
                None,
                flags=cv2.SOLVEPNP_SQPNP,
                iterationsCount=iterationsCount,
                reprojectionError=reprojectionError,
                confidence=confidence)
            if not success:
                return False, None
            r_pose = cv2.Rodrigues(r_pose)[0]  # world2cam == world2cam2
            RT = np.r_[np.c_[r_pose, t_pose], [(0, 0, 0, 1)]]  # world2cam2
            return True, np.linalg.inv(RT)  # cam2toworld
        elif len(pts2D) > 4 and mode == "poselib":
            assert HAS_POSELIB
            confidence = 0.9999
            iterationsCount = 10_000
            # NOTE: `Camera` struct currently contains `width`/`height` fields,
            # however these are not used anywhere in the code-base and are provided simply to be consistent with COLMAP.
            # so we put garbage in there
            colmap_intrinsics = opencv_to_colmap_intrinsics(K)
            fx = colmap_intrinsics[0, 0]
            fy = colmap_intrinsics[1, 1]
            cx = colmap_intrinsics[0, 2]
            cy = colmap_intrinsics[1, 2]
            width = img_size[0] if img_size is not None else int(cx * 2)
            height = img_size[1] if img_size is not None else int(cy * 2)

            if distortion is None:
                camera = {
                    'model': 'PINHOLE',
                    'width': width,
                    'height': height,
                    'params': [fx, fy, cx, cy]
                }
            else:
                camera = {
                    'model': 'OPENCV',
                    'width': width,
                    'height': height,
                    'params': [fx, fy, cx, cy] + distortion
                }

            pts2D = np.copy(pts2D)
            pts2D[:, 0] += 0.5
            pts2D[:, 1] += 0.5
            pose, _ = poselib.estimate_absolute_pose(
                pts2D, pts3D, camera, {
                    'max_reproj_error': reprojectionError,
                    'max_iterations': iterationsCount,
                    'success_prob': confidence
                }, {})
            if pose is None:
                return False, None
            RT = pose.Rt  # (3x4)
            RT = np.r_[RT, [(0, 0, 0, 1)]]  # world2cam
            return True, np.linalg.inv(RT)  # cam2toworld
        elif len(pts2D) > 4 and mode == "pycolmap":
            assert HAS_PYCOLMAP
            assert img_size is not None

            pts2D = np.copy(pts2D)
            pts2D[:, 0] += 0.5
            pts2D[:, 1] += 0.5
            colmap_intrinsics = opencv_to_colmap_intrinsics(K)
            fx = colmap_intrinsics[0, 0]
            fy = colmap_intrinsics[1, 1]
            cx = colmap_intrinsics[0, 2]
            cy = colmap_intrinsics[1, 2]
            width = img_size[0]
            height = img_size[1]
            if distortion is None:
                camera_dict = {
                    'model': 'PINHOLE',
                    'width': width,
                    'height': height,
                    'params': [fx, fy, cx, cy]
                }
            else:
                camera_dict = {
                    'model': 'OPENCV',
                    'width': width,
                    'height': height,
                    'params': [fx, fy, cx, cy] + distortion
                }

            pycolmap_camera = pycolmap.Camera(model=camera_dict['model'],
                                              width=camera_dict['width'],
                                              height=camera_dict['height'],
                                              params=camera_dict['params'])

            pycolmap_estimation_options = dict(
                ransac=dict(max_error=reprojectionError,
                            min_inlier_ratio=0.01,
                            min_num_trials=1000,
                            max_num_trials=100000,
                            confidence=0.9999))
            pycolmap_refinement_options = dict(refine_focal_length=False,
                                               refine_extra_params=False)
            ret = pycolmap.absolute_pose_estimation(
                pts2D,
                pts3D,
                pycolmap_camera,
                estimation_options=pycolmap_estimation_options,
                refinement_options=pycolmap_refinement_options)
            if ret is None:
                ret = {'success': False}
            else:
                ret['success'] = True
                if callable(ret['cam_from_world'].matrix):
                    retmat = ret['cam_from_world'].matrix()
                else:
                    retmat = ret['cam_from_world'].matrix
                ret['qvec'] = quaternion.from_rotation_matrix(retmat[:3, :3])
                ret['tvec'] = retmat[:3, 3]

            if not (ret['success'] and ret['num_inliers'] > 0):
                success = False
                pose = None
            else:
                success = True
                pr_world_to_querycam = np.r_[ret['cam_from_world'].matrix(),
                                             [(0, 0, 0, 1)]]
                pose = np.linalg.inv(pr_world_to_querycam)
            return success, pose
        else:
            return False, None
    except Exception as e:
        print(f'error during pnp: {e}')
        return False, None
