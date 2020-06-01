
import cv2
import numpy as np
from preprocess.preprocess import get_images
import os
import shutil
from skimage.transform import probabilistic_hough_line

def thinning (img):
     
    img1 = img.copy()
 
    # Structuring Element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    # Create an empty output image to hold values
    thin = np.zeros(img.shape,dtype='uint8')
 
    # Loop until erosion leads to an empty set
    while (cv2.countNonZero(img1)!=0):
        # Erosion
        erode = cv2.erode(img1,kernel)
        # Opening on eroded image
        opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel)
        # Subtract these two
        subset = erode - opening
        # Union of all previous sets
        thin = cv2.bitwise_or(subset,thin)
        # Set the eroded image for next iteration
        img1 = erode.copy()
    return thin

def skeleton(img):
    # Step 1: Create an empty skeleton
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    # Repeat steps 2-4
    while True:
        #Step 2: Open the image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv2.subtract(img, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(img)==0:
            break
    return skel
"""
def dilatation(image, dilatation_size, val_type):
    if val_type == 0:
        dilatation_type = cv2.MORPH_RECT
    elif val_type == 1:
        dilatation_type = cv2.MORPH_CROSS
    elif val_type == 2:
        dilatation_type = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dilatation_type, (dilatation_size, dilatation_size))
    dilatation_dst = cv2.dilate(image, element,iterations = 1)
    return dilatation_dst

def erosion(image,erosion_size, val_type):
    if val_type == 0:
        erosion_type = cv2.MORPH_RECT
    elif val_type == 1:
        erosion_type = cv2.MORPH_CROSS
    elif val_type == 2:
        erosion_type = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(erosion_type, (erosion_size , erosion_size))
    erosion_dst = cv2.erode(image, element,iterations = 1)
    return erosion_dst
"""


def houghline(input_path,output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    im_fn_list = get_images(input_path)
    for im_fn in im_fn_list:
        print(im_fn)
        #edge detection of the line
        
        im_raw = cv2.imread(im_fn)
        im_draw = im_raw.copy()
        im_gray = cv2.cvtColor(im_raw,cv2.COLOR_BGR2GRAY)

        im_gray = cv2.subtract(255, im_gray)
        im_gray = skeleton(im_gray)
        #im_gray = thinning(im_gray)
        cv2.imwrite(os.path.join(output_path, "skeleton_"+os.path.basename(im_fn)),im_gray)

        #im_gray = cv2.erode(im_gray, kernel,iterations = 1)
        #kernel1 = np.array([ [0, 0, 0],
        #            [0, 1, 0],
        #            [0, 0, 0]],np.uint8)

       # im_gray = cv2.morphologyEx(im_gray, cv2.MORPH_HITMISS, kernel1)
        #
        #kernel2 =  np.ones((2,2),np.uint8)
       # im_gray = cv2.erode(im_gray, kernel2,iterations = 1)
        #im_gray = cv2.morphologyEx(im_gray, cv2.MORPH_OPEN, kernel2)

 

        #im_gray = cv2.dilate(im_gray, element,iterations = 1)
        #cv2.imwrite(os.path.join(output_path, "diat_"+os.path.basename(im_fn)),im_gray)

        #im_gray = cv2.subtract(255, im_gray) 

        #im_gray = cv2.erode(im_gray, kernel2,iterations = 1)
        #cv2.imwrite(os.path.join(output_path, "erode_"+os.path.basename(im_fn)),im_gray)
        #im_gray = cv2.morphologyEx(im_gray, cv2.MORPH_OPEN, kernel1)
        #im_gray = cv2.dilate(im_gray, kernel1,iterations = 8)
        #cv2.imwrite(os.path.join(output_path, "diat_"+os.path.basename(im_fn)),im_gray)

        minLineLength = 50
        maxLineGap = 20
        """
        lines = probabilistic_hough_line(im_gray, threshold=100, line_length=minLineLength ,
                                 line_gap=maxLineGap)
        for line in lines:
            p0, p1 = line
            x1 = p0[0]
            y1 = p0[1]
            x2 = p1[0]
            y2 = p1[1]
            cv2.line(im_draw,(x1,y1),(x2,y2),(100),5)
        """
        lines = cv2.HoughLinesP(im_gray,1,0.1/180,100,minLineLength,maxLineGap)
        for iln in range(len(lines)):
            for x1,y1,x2,y2 in lines[iln]:
                cv2.line(im_draw,(x1,y1),(x2,y2),(100),5)
        cv2.imwrite(os.path.join(output_path, os.path.basename(im_fn)),im_draw)
