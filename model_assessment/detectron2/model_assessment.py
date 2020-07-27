#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/26 下午11:19
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm
# @Software: PyCharm
import argparse
import json
import time
import cv2
import torch
from detectron2.utils.logger import setup_logger

# from matplotlib import pyplot

# from google.colab.patches import cv2_imshow
setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import os


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default='/home/chase/projects/DeadChicken/data/models/detectron2/faster/faster_rcnn_R_50_FPN_1x.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='/home/chase/projects/DeadChicken/data/models/detectron2/faster/model_final.pth',
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
        default=0.9,
        type=float
    )
    parser.add_argument(
        '--draw_path',
        dest='draw_path',
        help='画图路径，如果不要画出来那就是None',
        default='/home/chase/projects/DeadChicken/data/models/detectron2/faster/output',
        type=str
    )

    return parser.parse_args()


def draw_bboxes(args, im, bboxes, name, type):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    COLOR = RED
    if type:
        COLOR = GREEN
    if bboxes != []:
        print(bboxes)
        font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
        for bbox in bboxes:
            cv2.rectangle(im, (bbox[1], bbox[2]), (bbox[3], bbox[4]), COLOR, thickness=2)
            # cv2.putText(im, bbox[0], (bbox[1], bbox[2]), font, 0.6, COLOR, 2)
    cv2.imwrite(os.path.join(args.draw_path, name), im)


def main(args):
    detectron2_repo_path = "/home/chase/projects/detectron2/"
    cfg = get_cfg()
    # cfg.merge_from_file("faster_rcnn_R_50_FPN_1x.yaml")
    cfg.merge_from_file(os.path.join(detectron2_repo_path, "configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 总共10个类别，不含背景
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128
    )  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # 模型阈值
    cfg.MODEL.WEIGHTS = "/home/chase/projects/detectron2/boyka/dead_chicken/output/model_final.pth"
    model = DefaultPredictor(cfg)
    with open(args.json, 'r') as fr:
        annotations = json.load(fr)
    print(annotations.keys())

    total_time = 0
    for image_info in annotations.get('images'):
        image = image_info.get('file_name')
        im = cv2.imread(os.path.join(args.srcImg, image))
        start_time=time.time()
        outputs = model(im)
        total_time+=time.time()-start_time

        pred_classes = outputs["instances"].pred_classes.cpu()
        pred_boxes = outputs["instances"].pred_boxes.tensor.cpu()
        pred_score = outputs["instances"].scores.cpu()

        result = torch.cat([torch.unsqueeze(pred_score, 1), pred_boxes], dim=-1).numpy()
        try:
            if args.draw_path != None:
                if not os.path.exists(args.draw_path):
                    os.makedirs(args.draw_path)
                draw_bboxes(args, im, result, image, False)
        except:
            print('验证的时候歇逼了')
        print(total_time)


if __name__ == '__main__':
    arg = parse_args()
    main(arg)
