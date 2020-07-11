# -*- coding: utf-8 -*-
# School                 ：UPC
# Author                 ：Boyka
# File Name              ：copy_val_images.py
# Computer User          ：Administrator 
# Current Project        ：DataProcess_Python
# Development Time       ：2020/2/22  15:38 
# Development Tool       ：PyCharm

import json
import os
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_image_path',
        dest='src_image_path',
        default='/home/chase/datasets/safety_helmet/images/',
        help='图片路径',
        type=str
    )
    parser.add_argument(
        '--tar_image_path',
        dest='tar_image_path',
        default='/home/chase/datasets/safety_helmet/test_images/',
        help='图片路径',
        type=str
    )
    parser.add_argument(
        '--annotation_path',
        dest='annotation_path',
        default='/home/chase/datasets/safety_helmet/safety_helmet_val.json',
        help='json路径',
        type=str
    )
    return parser.parse_args()


def file_copy(args_):
    if not os.path.exists(args_.tar_image_path):
        os.makedirs(args_.tar_image_path)
    annotation = json.load(open(args_.annotation_path))
    image_list = annotation.get("images")
    for image in image_list:
        shutil.copy(os.path.join(args_.src_image_path, image.get("file_name")),
                    os.path.join(args_.tar_image_path, image.get("file_name")))


if __name__ == '__main__':
    file_copy(parse_args())
