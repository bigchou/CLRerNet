# based on https://github.com/open-mmlab/mmdetection/blob/v2.28.0/demo/image_demo.py
# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet.apis import init_detector

from libs.api.inference import inference_one_image
from libs.utils.visualizer import visualize_lanes
from glob import glob
import os, copy, cv2
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default='result.png', help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold'
    )
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    #src, preds = inference_one_image(model, args.img)
    #import pdb
    #pdb.set_trace()
    # show the results
    #dst = visualize_lanes(src, preds, save_path=args.out_file)

    for i in glob('/home/iis/Desktop/CLRNet/images/*.jpg'):
    #for i in glob('/home/iis/Desktop/CLRNet/images1640590/*.jpg'):
    #for i in glob('/home/iis/Desktop/CLRNet/imageskeepap/*.jpg'):
        src, preds = inference_one_image(model, i)





        #dst = visualize_lanes(src, preds, save_path='output/%s'%(os.path.basename(i)))
        dst = copy.deepcopy(src)
        hits = [True for i in range(len(preds))]
        width=4
        PRED_HIT_COLOR = (0, 255, 0)
        PRED_MISS_COLOR = (0, 0, 255)

        for pred, hit in zip(preds, hits):
            color = PRED_HIT_COLOR if hit else PRED_MISS_COLOR
            #dst = draw_lane(pred, dst, dst.shape, width=4, color=color)
            if dst is None:
                dst = np.zeros(dst.shape, dtype=np.uint8)
            pred = pred.astype(np.int32)
            for p1, p2 in zip(pred[:-1], pred[1:]):
                cv2.line(dst, tuple(p1), tuple(p2), color, thickness=width)
        dst = cv2.resize(dst, (1920, 1080))
        save_path = 'output/%s'%(os.path.basename(i))
        cv2.imwrite(save_path, dst)
        print('save to ...', save_path)



# (clrernet) iis@iis-Z590-AORUS-ELITE-AX:~/Desktop/CLRerNet$
# python image_demo.py roundabout10_frame00025.jpg configs/clrernet/culane/clrernet_culane_dla34_ema.py clrernet_culane_dla34_ema.pth --out-file=result.png
if __name__ == '__main__':
    args = parse_args()
    main(args)
