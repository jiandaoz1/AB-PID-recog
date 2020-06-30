# coding=utf-8
import os
import shutil
import sys
import time

import cv2
import numpy as np
from preprocess.preprocess import get_images
def maskall(image_input_path,shapeloc_input_path,txtloc_input_path, output_path):
    """
    mask all shapes and text boxes
    """
    print("========== mask shapes and text ==============")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    im_fn_list = get_images(image_input_path)
    wl = 4
    for im_fn in im_fn_list:
        print('===============')
        print(im_fn)
        im_raw = cv2.imread(im_fn)
        im_gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY)[1]
        h, w = im_gray.shape[:2]
        im_blank =  np.ones(shape=[h, w], dtype=np.uint8)*255

        with open(os.path.join (shapeloc_input_path, os.path.splitext(os.path.basename(im_fn))[0]) + "_shapeloc.txt", 'r') as file1:
            for line in file1:
                cnt = [i for i in line.split()]
                loc = [int(i) for i in cnt[1:]]
                if cnt[0] == "CIR":
                    cv2.circle(im_blank, (loc[0],loc[1]), loc[2]+wl, 0, -1) 
                if cnt[0] == "IO":
                    cv2.drawContours(im_blank, [np.array(loc).reshape((-1,1,2))], 0, (0), thickness = -1, lineType=8)
        with open(os.path.join (txtloc_input_path, os.path.splitext(os.path.basename(im_fn))[0]) + "_txtloc.txt", 'r') as file2:
            for line in file2:
                if bool(line and line.strip()):
                    cnt = [i for i in line.split()]     
                    loc = [int(i) for i in cnt[1:-1]]
               
                    # crop image with rectangle box and save
                    x0,y0,w0,h0 = cv2.boundingRect(np.array(loc[:8]).astype(np.int32).reshape((-1, 2)))
                    img_crop = im_blank[y0:y0+h0,x0:x0+w0].copy()
                    hc, wc = img_crop.shape[:2]
                    countzero = hc*wc - cv2.countNonZero(img_crop)
                    if countzero == 0:
                        cv2.drawContours(im_blank, [np.array(loc[:8]).astype(np.int32).reshape((-1, 1, 2))], 0, (0), thickness=-1, lineType=8)
        #im_mask = cv2.bitwise_and(im_blank, im_gray, mask = None)
        im_mask = cv2.bitwise_or(cv2.bitwise_not(im_blank), im_gray, mask = None)
        cv2.imwrite(os.path.join(output_path, os.path.basename(im_fn)), im_mask)

