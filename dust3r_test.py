import os, sys
import mast3r.utils.path_to_dust3r  # noqa
paths = sys.path
for path in paths:
    print(path)
from test import run
from dust3r.model import AsymmetricCroCo3DStereo
from mast3r.model import AsymmetricMASt3R
if __name__ == '__main__':
    # device mac
    device = "cpu"
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"  # "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # 필요한 경우 로컬 체크포인트 경로를 model_name에 지정할 수 있습니다.
    try:
        model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    except Exception as e:
        model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    run(model, device)