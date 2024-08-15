from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
from typing import Any, List, Tuple, Dict, get_type_hints
import numpy as np
import torch

def infer_type_annotation(var: Any) -> str:
    """Infers the type annotation of a given variable."""
    if isinstance(var, list):
        if len(var) > 0:
            return f'List[{infer_type_annotation(var[0])}]'
        else:
            return 'List[Any]'
    elif isinstance(var, tuple):
        return f'Tuple[{", ".join(infer_type_annotation(v) for v in var)}]'
    elif isinstance(var, dict):
        if len(var) > 0:
            key_type = infer_type_annotation(next(iter(var.keys())))
            value_type = infer_type_annotation(next(iter(var.values())))
            return f'Dict[{key_type}, {value_type}]'
        else:
            return 'Dict[Any, Any]'
    elif isinstance(var, str):
        print("str[var]:", var)
        return 'str'
    elif isinstance(var, int):
        print("int[var]:", var)
        return 'int'
    elif isinstance(var, float):
        print("float[var]:", var)
        return 'float'
    elif isinstance(var, bool):
        print("bool[var]:", var)
        return 'bool'
    elif var is None:
        print("None[var]:", var)
        return 'None'
    else:
        if isinstance(var, np.ndarray):
            print("np.ndarray[var]:", var.shape)
            return 'np.ndarray'
        if isinstance(var, torch.Tensor):
            print("torch.Tensor[var]:", var.shape)
            return 'torch.Tensor'
        print("Any[var]:", var)
        return 'Any'


if __name__ == '__main__':
    device = 'cpu'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    images = load_images(['data/left_frames/0.png', 'data/right_frames/0.png'],
                         size=512)
    output: Dict[str, Dict[str, Any]] = inference([tuple(images)],
                       model,
                       device,
                       batch_size=1,
                       verbose=False)
    tensor_img = output["view1"]["img"] # (1, 3, 288, 512)
    # visualize tensor_img.
    from matplotlib import pyplot as plt
    plt.imshow(tensor_img.squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.show()


    """ output
    view1 (str): Dict (왼쪽, 호빈)
        img 
            tensor (1, 3, 288, 512) 
                (0,1) 쌍에서 view1의 이미지
        true_shape
            tensor (1, 2)
                - 처음 1: (0,1) 쌍에서 view1의 이미지 shape 
        idx
            list: [0]
                - view1이 0 index 
        instance
            list: ['0']
    view2 (str): Dict
        img 
            tensor (1, 3, 288, 512)
        true_shape
            tensor (1, 2)
        idx
            list: [1]
        instance
            list: ['1']
    pred1 (str): Dict
        pts3d
            tensor: (1, 288, 512, 3)
        conf
            tensor: (1, 288, 512)
        desc
            tensor: (1, 288, 512, 24)
        desc_conf
            tensor: (1, 288, 512)
    pred2 (str): Dict
        pts3d_in_other_view
            tensor: (1, 288, 512, 3)
        conf
            tensor: (1, 288, 512)
        desc
            tensor: (1, 288, 512, 24)
        desc_conf
            tensor: (1, 288, 512)
    loss (str): 
        - k1loss: v1: None
        

    """

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(
        0).detach()
    # desc1: (288, 512, 24)
    # desc2: (288, 512, 24)

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1,
                                                   desc2,
                                                   subsample_or_initxy1=8,
                                                   device=device,
                                                   dist='dot',
                                                   block_size=2**13)

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (
        matches_im0[:, 0] < int(W0) -
        3) & (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (
        matches_im1[:, 0] < int(W1) -
        3) & (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[
        valid_matches]

    # visualize a few matches
    import numpy as np
    import torch
    import torchvision.transforms.functional
    from matplotlib import pyplot as pl

    n_viz = 20
    num_matches = matches_im0.shape[0]
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1,
                                            n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[
        match_idx_to_viz], matches_im1[match_idx_to_viz]

    image_mean = torch.as_tensor([0.5, 0.5, 0.5],
                                 device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5],
                                device='cpu').reshape(1, 3, 1, 1)

    viz_imgs = []
    for i, view in enumerate([view1, view2]):
        rgb_tensor = view['img'] * image_std + image_mean
        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)),
                  'constant',
                  constant_values=0)
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)),
                  'constant',
                  constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1],
                '-+',
                color=cmap(i / (n_viz - 1)),
                scalex=False,
                scaley=False)
    pl.show(block=True)
