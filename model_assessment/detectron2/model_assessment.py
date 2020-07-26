#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/26 下午11:19
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm
# @Software: PyCharm
import cv2
# from matplotlib import pyplot
import matplotlib.pyplot as plt

from detectron2.utils.logger import setup_logger

# from google.colab.patches import cv2_imshow
setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os


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

im = cv2.imread("/home/chase/projects/DeadChicken/data/0.2/images/IMG20191029140938.jpg")
height, width = im.shape[0:2]
im = cv2.resize(im, (int(width / 5), int(height / 5)))
detectron2_repo_path = "/home/chase/projects/detectron2/"
cfg = get_cfg()
# cfg.merge_from_file("faster_rcnn_R_50_FPN_1x.yaml")
cfg.merge_from_file(os.path.join(detectron2_repo_path, "configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 总共10个类别，不含背景
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 模型阈值
cfg.MODEL.WEIGHTS = "/home/chase/projects/detectron2/boyka/dead_chicken/output/model_final.pth"
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

pred_classes = outputs["instances"].pred_classes
pred_boxes = outputs["instances"].pred_boxes
print(pred_classes, pred_boxes)

# 在原图上画出检测结果
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
plt.figure(2)
plt.imshow(v.get_image())
plt.show()
