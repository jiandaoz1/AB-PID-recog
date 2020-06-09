
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


def houghline_detect(input_path,output_path):
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

        minLineLength = 5
        maxLineGap = 1
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
        lines = cv2.HoughLinesP(im_gray,1,1/180,100,minLineLength,maxLineGap)
        for iln in range(len(lines)):
            for x1,y1,x2,y2 in lines[iln]:
                cv2.line(im_draw,(x1,y1),(x2,y2),(100),5)

        cv2.imwrite(os.path.join(output_path, os.path.basename(im_fn)),im_draw)
def scale_contour(cnt, scale):
    """
    re-scale the contour
    """
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

def inoutlet_detect(input_path,output_path, mask):
    """
    Identify the inlet and outlet arrow.
    """
    epsilon = 0.02
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    im_fn_list = get_images(input_path)
    for im_fn in im_fn_list:
        print('===============')
        print(im_fn)
        im_raw = cv2.imread(im_fn)

        im_gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY)[1]
        im_gray = cv2.subtract(255, im_gray)

        h, w = im_gray.shape[:2]
        im_blank =  np.ones(shape=[h, w], dtype=np.uint8)*255

        _, contours, _ = cv2.findContours(im_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
           approx = cv2.approxPolyDP(cnt,epsilon*cv2.arcLength(cnt,True),True)
           #cv2.drawContours(im_raw, [approx], 0, (0,255,0), thickness = 1, lineType=8)
           # Identify all contours with only 5 or 7 vertices and contour area is greater than 700
           if (len(approx)==7 or len(approx)==5) and cv2.contourArea(cnt)>700:
               l = len(approx)
               # Iterate all vertices as a arrow tip point
               for it  in range(len(approx)):
                   tip = approx[it][0]
                   othervert = np.delete(approx, it, 0)
                   sumpts = np.zeros(2)
                   for ivert in othervert:
                       sumpts = sumpts + ivert
                   # Calculate the centroid for the rest of symmetric point
                   centpts = sumpts[0]/(l-1)
                   A1 = tip[1]-centpts[1]
                   B1 = -(tip[0]-centpts[0])
                   dis = []
                   for ipt in range(1,int((l+1)/2)):
                       il1 = it + ipt
                       if il1 > l-1:
                           il1 = il1 - l
                       il2 = it - ipt
                       A2 = approx[il2][0][1] - approx[il1][0][1]
                       B2 = -(approx[il2][0][0] - approx[il1][0][0])
                       # Calculate the distance between each of symmetric points 
                       dis.append(np.sqrt(A2**2+B2**2))
                       # Calculate the intersection angle between the line across tip and centroid and the line across two symmetric points
                       theta = abs(np.arccos((A1*A2+B1*B2)/np.sqrt((A1**2+B1**2)*(A2**2+B2**2))))
                       # Check if the line across tip and centroid is perpendicular with the line across two symmetric points with +/- 5 degree accuracy. 
                       # Check if the distance between two symmetric points keep constant or decreasing with +5 pixel accuracy
                       if theta < 85/180*np.pi or theta > 95/180*np.pi or (ipt>1 and dis[ipt-1] - dis[ipt-2] > 5):
                           break
                       if ipt == int((l+1)/2)-1:
                           for item in othervert:
                               cv2.drawMarker(im_raw, (item[0][0], item[0][1]),(0,255,0), markerType=cv2.MARKER_STAR, 
                                markerSize=4, thickness=1, line_type=cv2.LINE_AA)
                           cv2.drawMarker(im_raw, (tip[0], tip[1]),(0,0,255), markerType=cv2.MARKER_STAR, 
                                markerSize=8, thickness=1, line_type=cv2.LINE_AA)
                           cv2.drawMarker(im_raw, (int(centpts[0]), int(centpts[1])),(255,0,0), markerType=cv2.MARKER_STAR, 
                                markerSize=8, thickness=1, line_type=cv2.LINE_AA)
                           approx_scaled = scale_contour(approx,1.1)
                           cv2.drawContours(im_blank, [approx_scaled], 0, (0), thickness = -1, lineType=8)
                   im_woarrow = cv2.bitwise_or (cv2.bitwise_not(im_blank), cv2.bitwise_not(im_gray), mask = None)
        if mask == True:
            cv2.imwrite(os.path.join(output_path, os.path.basename(im_fn)), im_woarrow)
        else:
            cv2.imwrite(os.path.join(output_path, os.path.basename(im_fn)), im_raw)



def square_detect(input_path,output_path):
    """
    Identify the square which is typically intrument alarm.
    """
    epsilon = 0.001
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    im_fn_list = get_images(input_path)
    for im_fn in im_fn_list:
        print('===============')
        print(im_fn)
        im_raw = cv2.imread(im_fn)

        im_gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY)[1]
        #im_gray = cv2.subtract(255, im_gray)

        _, contours, _ = cv2.findContours(im_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt,epsilon*cv2.arcLength(cnt,True),True)
            #for item in approx:
              #  cv2.drawMarker(im_gray, (item[0][0], item[0][1]),(0,255,0), markerType=cv2.MARKER_STAR,markerSize=4, thickness=1, line_type=cv2.LINE_AA)          
            # Identify all contours with only 5 or 7 vertices and contour area is greater than 700
            if (len(approx)==4) and cv2.contourArea(cnt)>400:
                cv2.drawContours(im_raw, [approx], 0, (0,255,0), thickness = 3, lineType=8)

                dis = []
                A1 = approx[3][0][1] - approx[1][0][1]
                B1 = -(approx[3][0][0] - approx[1][0][0])
                dis.append(np.sqrt(A1**2+B1**2))
                A2 = approx[2][0][1] - approx[0][0][1]
                B2 = -(approx[2][0][0] - approx[0][0][0])
                # Calculate the distance between two vertex
                dis.append(np.sqrt(A2**2+B2**2))
                # Calculate the intersection angle between the line across tip and centroid and the line across two symmetric points
                theta = abs(np.arccos((A1*A2+B1*B2)/np.sqrt((A1**2+B1**2)*(A2**2+B2**2))))
                # Check if the line across vertex perpendicular with each other. 
                # Check if the distance between two vertex points are the same
                if theta > 85/180*np.pi and theta < 95/180*np.pi and np.abs(dis[0] - dis[1]) < 5:
                    for item in approx:
                        cv2.drawMarker(im_raw, (item[0][0], item[0][1]),(0,255,0), markerType=cv2.MARKER_STAR,markerSize=4, thickness=1, line_type=cv2.LINE_AA)
      
        cv2.imwrite(os.path.join(output_path, os.path.basename(im_fn)), im_raw)

def circle_detect(input_path,output_path, mask):
    """
    Identify the circles using houghcircle approach
    """

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    im_fn_list = get_images(input_path)
    for im_fn in im_fn_list:
        print('===============')
        print(im_fn)
        im_raw = cv2.imread(im_fn)
        im_gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
        #im_gray= cv2.subtract(255, im_gray)

        #cv2.imwrite(os.path.join(output_path, os.path.basename(im_fn)), im_gray)
        # apply automatic Canny edge detection using the computed median
        circles = cv2.HoughCircles(im_gray,cv2.HOUGH_GRADIENT,1.8,100,  
                                   param1=200, param2=80,minRadius=9,maxRadius=60)

        h, w = im_gray.shape[:2]
        im_blank =  np.ones(shape=[h, w], dtype=np.uint8)*255
        im_blank2 = im_blank.copy()
        # pixel width of each circle
        wl =4

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:

                cv2.circle(im_blank, (i[0],i[1]), int(i[2]+wl), 0, -1)  # -1 to draw filled circles
                cv2.circle(im_blank, (i[0],i[1]), int(i[2]-wl), 255, -1)  # -1 to draw filled circles
                
                #cv2.imwrite(os.path.join(output_path, "cir"+"-"+os.path.basename(im_fn)), im_blank) 

            bit_or = cv2.bitwise_or(im_blank, im_gray, mask = None)
            bit_or = cv2.threshold(bit_or, 128, 255, cv2.THRESH_BINARY)[1]
            bit_or= cv2.subtract(255, bit_or)
            for i in  circles[0,:]:
                circheck = []
                for iy in range(i[1]-i[2]-wl,i[1]+i[2]+wl):
                    countblkpixel = np.sum(bit_or[iy,i[0]-i[2]-wl:i[0]+i[2]+wl])
                    if countblkpixel > 2:
                        circheck.append(1)
                    else:
                        circheck.append(0)
                # check if each ring contain a circle
                if np.sum(circheck)*1.0/len(circheck) > 0.9:
                    # draw the outer circle
                    cv2.circle(im_raw,(i[0],i[1]),i[2],(0,255,0),2)                
                    # draw the center of the circle
                    cv2.circle(im_raw,(i[0],i[1]),2,(0,0,255),3)
                    cv2.circle(im_blank2, (i[0],i[1]), int(i[2]+wl), 0, -1)  # -1 to draw filled circles
                    im_wocircle = cv2.bitwise_or(cv2.bitwise_not(im_blank2), im_gray, mask = None)
        if mask == True:
            cv2.imwrite(os.path.join(output_path, os.path.basename(im_fn)), im_wocircle)
        else:
            cv2.imwrite(os.path.join(output_path, os.path.basename(im_fn)), im_raw)

