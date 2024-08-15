# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed to load image pairs
# --------------------------------------------------------
import numpy as np
import torch

from typing import List, Tuple, Dict, Any, Set


def make_pairs(
        imgs: List[Dict[str, Any]],
        scene_graph: str = 'complete',
        prefilter: str = None,
        symmetrize: bool = True) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
1. **입력 인자**:
   - `imgs`: 이미지들의 리스트.
   - `scene_graph`:
        이미지 쌍을 생성할 규칙을 정의하는 문자열.
        예를 들어, 'complete', 'swin', 'logwin', 'oneref' 등이 있습니다.
   - `symmetrize`:
        쌍을 대칭으로 만들지 여부를 결정하는 부울 값.

### 주요 로직

1. **complete 그래프**:
   - 모든 이미지 쌍을 생성합니다.
    - 이미지 i와 j에 대해 i < j 조건을 만족하는 모든 쌍을 만듭니다.

5. **대칭화** (`symmetrize`):
   - `symmetrize`가 True인 경우, 생성된 쌍을 대칭으로 만듭니다.
   - 즉, (A, B) 쌍이 있다면 (B, A) 쌍도 추가합니다.

- **완전 그래프 (complete)**:
  - 주어진 이미지 리스트에서 가능한 모든 쌍을 생성하여 매칭 작업에 사용할 수 있음
    """
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    if scene_graph == 'complete':  # complete graph
        for i in range(len(imgs)):
            for j in range(i):
                # (imgs[i], imgs[j]): Tuple[Dict[str, Any], Dict[str, Any]]
                pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith('swin'):
        iscyclic: bool = not scene_graph.endswith('noncyclic')
        try:
            winsize: int = int(scene_graph.split('-')[1])
        except Exception as e:
            winsize = 3
        pairsid: Set[Tuple[int, int]] = set()
        for i in range(len(imgs)):
            for j in range(1, winsize + 1):
                idx: int = (i + j)
                if iscyclic:
                    idx = idx % len(imgs)  # explicit loop closure
                if idx >= len(imgs):
                    continue
                pairsid.add((i, idx) if i < idx else (idx, i))
        for i, j in pairsid:
            pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith('logwin'):
        iscyclic: bool = not scene_graph.endswith('noncyclic')
        try:
            winsize: int = int(scene_graph.split('-')[1])
        except Exception as e:
            winsize = 3
        offsets: List[int] = [2**i for i in range(winsize)]
        pairsid: Set[Tuple[int, int]] = set()
        for i in range(len(imgs)):
            ixs_l: List[int] = [i - off for off in offsets]
            ixs_r: List[int] = [i + off for off in offsets]
            for j in ixs_l + ixs_r:
                if iscyclic:
                    j = j % len(imgs)  # Explicit loop closure
                if j < 0 or j >= len(imgs) or j == i:
                    continue
                pairsid.add((i, j) if i < j else (j, i))
        for i, j in pairsid:
            pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith('oneref'):
        refid: int = int(scene_graph.split('-')[1]) if '-' in scene_graph else 0
        for j in range(len(imgs)):
            if j != refid:
                pairs.append((imgs[refid], imgs[j]))
    if symmetrize:
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]
        pairs += [(img2, img1) for img1, img2 in pairs]

    # now, remove edges
    if isinstance(prefilter, str) and prefilter.startswith('seq'):
        pairs = filter_pairs_seq(pairs, int(prefilter[3:]))

    if isinstance(prefilter, str) and prefilter.startswith('cyc'):
        pairs = filter_pairs_seq(pairs, int(prefilter[3:]), cyclic=True)

    return pairs


def filter_pairs_seq(
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        seq_dis_thr: int,
        cyclic: bool = False) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    edges: List[Tuple[int, int]] = [
        (img1['idx'], img2['idx']) for img1, img2 in pairs
    ]
    kept: List[int] = _filter_edges_seq(edges, seq_dis_thr, cyclic=cyclic)
    return [pairs[i] for i in kept]


def _filter_edges_seq(edges: List[Tuple[int, int]],
                      seq_dis_thr: int,
                      cyclic: bool = False) -> List[int]:
    # number of images
    n: int = max(max(e) for e in edges) + 1

    kept: List[int] = []
    for e, (i, j) in enumerate(edges):
        dis: int = abs(i - j)
        if cyclic:
            dis = min(dis, abs(i + n - j), abs(i - n - j))
        if dis <= seq_dis_thr:
            kept.append(e)
    return kept


def sel(x, kept):
    if isinstance(x, dict):
        return {k: sel(v, kept) for k, v in x.items()}
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return x[kept]
    if isinstance(x, (tuple, list)):
        return type(x)([x[k] for k in kept])


def _filter_edges_seq(edges, seq_dis_thr, cyclic=False):
    # number of images
    n = max(max(e) for e in edges) + 1

    kept = []
    for e, (i, j) in enumerate(edges):
        dis = abs(i - j)
        if cyclic:
            dis = min(dis, abs(i + n - j), abs(i - n - j))
        if dis <= seq_dis_thr:
            kept.append(e)
    return kept


def filter_pairs_seq(pairs, seq_dis_thr, cyclic=False):
    edges = [(img1['idx'], img2['idx']) for img1, img2 in pairs]
    kept = _filter_edges_seq(edges, seq_dis_thr, cyclic=cyclic)
    return [pairs[i] for i in kept]


def filter_edges_seq(view1, view2, pred1, pred2, seq_dis_thr, cyclic=False):
    edges = [(int(i), int(j)) for i, j in zip(view1['idx'], view2['idx'])]
    kept = _filter_edges_seq(edges, seq_dis_thr, cyclic=cyclic)
    print(
        f'>> Filtering edges more than {seq_dis_thr} frames apart: kept {len(kept)}/{len(edges)} edges'
    )
    return sel(view1, kept), sel(view2, kept), sel(pred1,
                                                   kept), sel(pred2, kept)
