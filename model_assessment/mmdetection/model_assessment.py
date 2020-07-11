#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/11 下午10:29
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm
import argparse
import json
import os
import time

import cv2
import numpy as np
from mmdet.apis import inference_detector
from mmdet.apis import init_detector


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default='/home/chase/projects/DeadChicken/data/models/mmdetecion/ssd_300/ssd300.py',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='/home/chase/projects/DeadChicken/data/models/mmdetecion/ssd_300/latest.pth',
        type=str
    )
    parser.add_argument(
        '--srcImg',
        dest='srcImg',
        help='(/path/to/test/images)',
        default='/home/chase/projects/DeadChicken/data/0.2/images',
        type=str
    )
    parser.add_argument(
        '--json',
        dest='json',
        help='The location of jsons ',
        default='/home/chase/projects/DeadChicken/data/0.2/dead_chicken_valid.json',
        type=str
    )
    parser.add_argument(
        '--classes',
        dest='classes',
        help='The location of annotations',
        default=['DeadLayer'],
        type=list
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='交幷比阈值',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--draw_path',
        dest='draw_path',
        help='画图路径，如果不要画出来那就是None',
        default='/home/chase/projects/DeadChicken/data/models/mmdetecion/ssd_300/output',
        type=str
    )

    return parser.parse_args()


def draw_bboxes(args, im, bboxes, name, type):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    COLOR = RED
    if type:
        COLOR = GREEN
        # print('gt')
    if bboxes != []:
        print(bboxes)
        font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
        for bbox in bboxes:
            cv2.rectangle(im, (bbox[2], bbox[3]), (bbox[4], bbox[5]), COLOR, thickness=2)
            cv2.putText(im, bbox[0], (bbox[2], bbox[3]), font, 0.6, COLOR, 2)
    cv2.imwrite(os.path.join(args.draw_path, name), im)


def main(args):
    model = init_detector(args.cfg, args.weights, device='cuda:0')
    with open(args.json, 'r') as fr:
        annotations = json.load(fr)
    print(annotations.keys())

    total_time = 0
    for image_info in annotations.get('images'):
        image = image_info.get('file_name')

        im = cv2.imread(os.path.join(args.srcImg, image))
        start_time = time.time()
        result = inference_detector(model, im)
        total_time+=time.time()-start_time
        print(total_time)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # print(bboxes)
        # print(labels)
        result = []

        for bbox, label in zip(bboxes, labels):
            result_bbox = []
            if bbox[4] < 0.9:
                continue
            result_bbox.append(args.classes[label])
            result_bbox.append(bbox[4])
            result_bbox.extend(bbox[0:4])
            result.append(result_bbox)
        print(result)
        try:
            if args.draw_path != None:
                if not os.path.exists(args.draw_path):
                    os.makedirs(args.draw_path)
                draw_bboxes(args, im, result, image, False)
        except:
            print('验证的时候歇逼了')


if __name__ == '__main__':
    arg = parse_args()
    main(arg)
