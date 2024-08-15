# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dust3r submodule import
# --------------------------------------------------------

import sys
import os.path as path

"""
__file__: 현재 실행 중인 Python 스크립트의 파일 경로를 나타냅니다.
    - /Users/user/PycharmProjects/mast3r/mast3r/utils/path_to_dust3r.py
path.dirname(__file__): 
    - 현재 파일의 디렉토리 경로를 반환
    - /Users/user/PycharmProjects/mast3r/mast3r/utils
HERE_PATH = path.normpath(): 
    - 주어진 경로를 표준화하여, 운영체제에 따라 적절한 경로 형식을 갖추게 합니다. 
    - 예를 들어, Windows에서는 백슬래시(\), 유닉스 기반 시스템에서는 슬래시(/)를 사용합니다.
    - /Users/user/PycharmProjects/mast3r/mast3r/utils
"""
HERE_PATH = path.normpath(path.dirname(__file__))
# DUSt3R_REPO_PATH: /Users/user/PycharmProjects/mast3r/dust3r
DUSt3R_REPO_PATH = path.normpath(path.join(HERE_PATH, '../../dust3r'))
# DUSt3R_LIB_PATH: /Users/user/PycharmProjects/mast3r/dust3r/dust3r
DUSt3R_LIB_PATH = path.join(DUSt3R_REPO_PATH, 'dust3r')
# check the presence of models directory in repo to be sure its cloned
if path.isdir(DUSt3R_LIB_PATH):
    # workaround for sibling import
    sys.path.insert(0, DUSt3R_REPO_PATH)
else:
    raise ImportError(
        f"dust3r is not initialized, could not find: {DUSt3R_LIB_PATH}.\n "
        "Did you forget to run 'git submodule update --init --recursive' ?")
