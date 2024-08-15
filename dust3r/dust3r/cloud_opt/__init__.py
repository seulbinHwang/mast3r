# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# global alignment optimization wrapper function
# --------------------------------------------------------
from enum import Enum

from .optimizer import PointCloudOptimizer
from .modular_optimizer import ModularPointCloudOptimizer
from .pair_viewer import PairViewer
from .base_opt import BasePCOptimizer
"""
### `GlobalAlignerMode` Enum

#### 목적
`GlobalAlignerMode` 열거형 클래스는 
    글로벌 정렬(align) 최적화 작업을 수행하기 위해 사용할 최적화기의 종류를 정의

#### 내용
- **PointCloudOptimizer**: 전체 포인트 클라우드 최적화를 수행하는 클래스.
- **ModularPointCloudOptimizer**: 모듈식 포인트 클라우드 최적화를 수행하는 클래스.
- **PairViewer**: 두 이미지 쌍의 시각화를 위해 더미 최적화 도구로 사용되는 클래스.


"""


class GlobalAlignerMode(Enum):
    PointCloudOptimizer = "PointCloudOptimizer"
    ModularPointCloudOptimizer = "ModularPointCloudOptimizer"
    PairViewer = "PairViewer"


def global_aligner(dust3r_output,
                   device,
                   mode=GlobalAlignerMode.PointCloudOptimizer,
                   **optim_kw) -> BasePCOptimizer:
    """
#### 목적
`global_aligner` 함수는 DUST3R의 출력을 받아 지정된 모드에 따라 적절한 최적화기를 생성하고 반환
    이 함수는 글로벌 정렬 작업을 쉽게 설정할 수 있도록 하는 래퍼(wrapper) 함수

#### 내용

1. **입력 파라미터**:
   - `dust3r_output`: DUST3R 모델의 출력 결과를 포함하는 딕셔너리.
   - `device`: 최적화 작업을 수행할 장치 (예: 'cuda' 또는 'cpu').
   - `mode`:
        사용할 최적화기의 종류를 지정하는 `GlobalAlignerMode` 열거형 값.
        기본값은 `PointCloudOptimizer`.
   - `optim_kw`: 최적화기 생성 시 추가로 전달할 키워드 인자들.
    """
    # extract all inputs
    view1, view2, pred1, pred2 = [
        dust3r_output[k] for k in 'view1 view2 pred1 pred2'.split()
    ]
    # build the optimizer
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        net = PointCloudOptimizer(view1, view2, pred1, pred2,
                                  **optim_kw).to(device)
    elif mode == GlobalAlignerMode.ModularPointCloudOptimizer:
        net = ModularPointCloudOptimizer(view1, view2, pred1, pred2,
                                         **optim_kw).to(device)
    elif mode == GlobalAlignerMode.PairViewer:
        net = PairViewer(view1, view2, pred1, pred2, **optim_kw).to(device)
    else:
        raise NotImplementedError(f'Unknown mode {mode}')

    return net
