#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/11 下午12:56
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm

import os

import cv2

src_path = '/home/chase/projects/DeadChicken/data/Images/Images'
tar_path = '/home/chase/projects/DeadChicken/data/0.2/images'
if not os.path.exists(tar_path):
    os.mkdir(tar_path)
for img in os.listdir(src_path):
    cv_img = cv2.imread(os.path.join(src_path, img))
    height, width = cv_img.shape[0:2]
    cv_img = cv2.resize(cv_img, (int(width / 5), int(height / 5)))
    cv2.imwrite(os.path.join(tar_path, img), cv_img)
