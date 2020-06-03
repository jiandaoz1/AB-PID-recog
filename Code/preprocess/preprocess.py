
import os
import shutil
import sys
import time

import cv2
import numpy as np
import tensorflow as tf


def get_images(input_path):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']

    for parent, dirnames, filenames in os.walk(input_path):

        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(2400) / float(im_size_min)
    print("scale factor = "+str(im_scale))
    if np.round(im_scale * im_size_max) > 3600:
        im_scale = float(2400) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)
   
    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16
    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.cv2.INTER_CUBIC)
    return re_im, (new_h / img_size[0], new_w / img_size[1])

def remove_border(input_path, output_path):
    """remove P&ID drawing broder
    """

    im_fn_list = get_images(input_path)
    epsilon = 0.0001
    for im_fn in im_fn_list:
        print('===============')
        print(im_fn)
        start = time.time()
        try:
            im_gray = cv2.imread(im_fn,cv2.IMREAD_GRAYSCALE)
        except:
            print("Error reading image {}!".format(im_fn))
            continue
        h, w = im_gray.shape[:2]
        im_blank =  np.ones(shape=[h, w], dtype=np.uint8)*255
        im_binary = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY)[1]
        _, contours, _ = cv2.findContours(im_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        area = []
        for cnt in contours:
           approx = cv2.approxPolyDP(cnt,epsilon*cv2.arcLength(cnt,True),True)
           area .append(cv2.contourArea(cnt))
        # sort by contour area
        top_cnt_area = np.argsort(-1*np.array(area))
        # drawing has not been pre-processed
        # select the thrid largest contour which fit the drawing broader
        ind = top_cnt_area[2]
        approx = cv2.approxPolyDP(contours[ind],epsilon*cv2.arcLength(contours[ind],True),True)
        cv2.drawContours(im_blank, [approx], 0, (0), thickness = -1, lineType=8)
        # combine image with masks
        im_gray = cv2.bitwise_or(im_blank, im_gray)
        cv2.imwrite(os.path.join(output_path, os.path.basename(im_fn)), im_gray)
        
