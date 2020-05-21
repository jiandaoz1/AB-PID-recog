
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
            im_grey = cv2.imread(im_fn,cv2.IMREAD_GRAYSCALE)
        except:
            print("Error reading image {}!".format(im_fn))
            continue
        h, w = im_grey.shape[:2]
        im_blank =  np.zeros(shape=[h, w], dtype=np.uint8)
        im_binary = cv2.threshold(im_grey, 128, 255, cv2.THRESH_BINARY)[1]
        _, contours, _ = cv2.findContours(im_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        area = []
        for cnt in contours:
           approx = cv2.approxPolyDP(cnt,epsilon*cv2.arcLength(cnt,True),True)
           area .append(cv2.contourArea(cnt))
        # sort by contour area
        top_cnt_area = np.argsort(-1*np.array(area))
        # select the thrid largest contour which fit the drawing broader
        ind = top_cnt_area[2]
        approx = cv2.approxPolyDP(contours[ind],epsilon*cv2.arcLength(contours[ind],True),True)
        cv2.drawContours(im_blank, [approx], 0, (255), thickness = -1, lineType=8)
        # combine image with masks
        im_grey = cv2.bitwise_and(im_blank, im_grey)
        
        """
        #edge detection of the line
        #edges = cv2.Canny(im_grey,50,150,apertureSize = 3)
        minLineLength = 10
        maxLineGap = 50
        lines = cv2.HoughLines(im_grey,1,np.pi/180,100,minLineLength,maxLineGap)
        for iln in range(len(lines)):
            for x1,y1,x2,y2 in lines[iln]:
                print(x1,y1,x2,y2)
                cv2.line(im_grey,(x1,y1),(x2,y2),(100),15)
        """
        cv2.imwrite(os.path.join(output_path, os.path.basename(im_fn)), im_grey)
        
