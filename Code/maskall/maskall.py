# coding=utf-8
import os
import shutil
import sys
import time

import cv2
import numpy as np
from preprocess.preprocess import get_images
def maskall(image_input_path,shapeloc_input_path,txtloc_input_path, comploc_input_path, output_path):
    """
    mask all shapes and text boxes
    """
    print("========== mask shapes and text ==============")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    img_fn_list = get_images(image_input_path)
    wl = 4
    for img_fn in img_fn_list:
        print(img_fn)
        img_raw = cv2.imread(img_fn)
        img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        img_binary = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)[1]
        h, w = img_binary.shape[:2]
        img_blank =  np.ones(shape=[h, w], dtype=np.uint8)*255

        with open(os.path.join (shapeloc_input_path, os.path.splitext(os.path.basename(img_fn))[0]) + "_loc.txt", 'r') as file1:
            for line in file1:
                cnt = [i for i in line.split()]
                loc = [int(i) for i in cnt[1:]]
                cv2.drawContours(img_blank, [np.array(loc).reshape((-1,1,2))], 0, (0), thickness = -1, lineType=8)

        with open(os.path.join (comploc_input_path, os.path.splitext(os.path.basename(img_fn))[0]) + "_loc.txt", 'r') as file2:
            for line in file2:
                cnt = [i for i in line.split()]
                loc = [int(i) for i in cnt[1:]]
                cv2.drawContours(img_blank, [np.array(loc).reshape((-1,1,2))], 0, (0), thickness = -1, lineType=8)

        with open(os.path.join (txtloc_input_path, os.path.splitext(os.path.basename(img_fn))[0]) + "_loc.txt", 'r') as file3:
            for line in file3:
                if bool(line and line.strip()):
                    cnt = [i for i in line.split()]     
                    loc = [int(i) for i in cnt[1:]]
                    # crop image with rectangle box and save
                    x0,y0,w0,h0 = cv2.boundingRect(np.array(loc[:8]).astype(np.int32).reshape((-1, 2)))
                    img_crop = img_blank[y0:y0+h0,x0:x0+w0].copy()
                    hc, wc = img_crop.shape[:2]
                    countzero = hc*wc - cv2.countNonZero(img_crop)
                    if countzero *1.0 / (hc*wc) <= 0.25:
                        # if new area is less than 25% of overlap with other proposed masked area
                        cv2.drawContours(img_blank, [np.array(loc[:8]).astype(np.int32).reshape((-1, 1, 2))], 0, (0), thickness=-1, lineType=8)
 
                        #img_mask = cv2.bitwise_and(img_blank, img_binary, mask = None)
        img_mask = cv2.bitwise_or(cv2.bitwise_not(img_blank), img_binary, mask = None)
        cv2.imwrite(os.path.join(output_path, os.path.basename(img_fn)), img_mask)

