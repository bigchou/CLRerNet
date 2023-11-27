# modified based on:
# https://github.com/open-mmlab/mmdetection/blob/v2.28.0/mmdet/apis/inference.py
# Copyright (c) OpenMMLab. All rights reserved.

import cv2
import torch
from mmcv.parallel import collate, scatter

from libs.datasets.pipelines import Compose
from libs.datasets.metrics.culane_metric import interp

import pdb, json, os
import numpy as np

def inference_one_image(model, img_path):
    """Inference on an image with the detector.
    Args:
        model (nn.Module): The loaded detector.
        img_path (str): Image path.
    Returns:
        img (np.ndarray): Image data with shape (width, height, channel).
        preds (List[np.ndarray]): Detected lanes.
    """
    img = cv2.imread(img_path)
    #img = cv2.resize(img, (1640, 590)) # <-------------------------------------------------
    ori_shape = img.shape
    print(ori_shape)
    data = dict(
        filename=img_path,
        sub_img_name=None,
        img=img,
        gt_points=[],
        id_classes=[],
        id_instances=[],
        img_shape=ori_shape,
        ori_shape=ori_shape,
    )

    cfg = model.cfg
    model.bbox_head.test_cfg.as_lanes = False
    device = next(model.parameters()).device  # model device


    with open('crop2.json') as f: jsondata = json.load(f)
    key_list = []
    value_list = []
    for i in jsondata:
        key_list.append(i)
        value_list.append(jsondata[i])
    for i, v in enumerate(key_list):
        if v in os.path.basename(img_path):
            crop = value_list[i]
    print("crop:", crop)

    cfg.data.test.pipeline[0]['pipelines'][1]['y_min'] = crop # <----------------
    model.test_cfg['cut_height'] = crop # <--------------------------------------

    print(cfg.data.test.pipeline[0])
    print(model.test_cfg)

    #pdb.set_trace()
    '''
    print(
        type(cfg.data.test.pipeline[0]),
        cfg.data.test.pipeline[0]
    )
    '''

    test_pipeline = Compose(cfg.data.test.pipeline)

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    data['img_metas'] = data['img_metas'].data[0]
    data['img'] = data['img'].data[0]

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # type(data) = dict
    # data.keys() = dict_keys(['img_metas', 'img'])
    # data['img'].shape = torch.Size([1, 3, 320, 800])
    
    # forward the model
    sample_img = (data['img'].permute(2,3,1,0).squeeze()*255).cpu().numpy().astype(np.uint8)
    cv2.imwrite("sfsfwerwrwrwrqwr.png", sample_img)
    
    
    
    data['img_metas'][0]['filename'] = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa123.jpg'
    #pdb.set_trace()
    with torch.no_grad():
        # (Pdb) data['img_metas']
        # [{
        # 'filename': '/home/iis/Desktop/CLRNet/images1640590/roundabout4_frame00150.jpg',
        # 'sub_img_name': None,
        # 'ori_shape': (590, 1640, 3),
        # 'img_shape': (320, 800, 3),
        # 'img_norm_cfg': {'mean': array([0., 0., 0.], dtype=float32), 'std': array([255., 255., 255.], dtype=float32),
        # 'to_rgb': False}}]
        results = model(return_loss=False, rescale=True, **data)

    lanes = results[0]['result']['lanes']
    preds = get_prediction(lanes, ori_shape[0], ori_shape[1])
    #pdb.set_trace()

    return img, preds


def get_prediction(lanes, ori_h, ori_w):
    preds = []
    for lane in lanes:
        lane = lane.cpu().numpy()
        xs = lane[:, 0]
        ys = lane[:, 1]
        valid_mask = (xs >= 0) & (xs < 1)
        xs = xs * ori_w
        lane_xs = xs[valid_mask]
        lane_ys = ys[valid_mask] * ori_h
        lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
        pred = [(x, y) for x, y in zip(lane_xs, lane_ys)]
        interp_pred = interp(pred, n=5)
        preds.append(interp_pred)
    return preds
