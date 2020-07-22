import cv2
import os
import shutil
import numpy as np
from preprocess.preprocess import get_images
from matplotlib import pyplot as plt
from utils.utils import  txtremove
import imutils




def compmatch(input_path,template_path, output_path,comploc_output_path):

    """
    Component Match by using template matching
    """

    print("========== Component Match ==============")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    img_fn_list = get_images(input_path)
    tar_fn_list = get_images(template_path)
    threshold = 0.8
    for img_fn in img_fn_list:
        print(img_fn)

        txtfileloc = os.path.join (comploc_output_path, os.path.splitext(os.path.basename(img_fn))[0]) + "_loc.txt" 

        img_gray = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)
        h,w = img_gray.shape
        img_blank =  np.ones(shape=[h, w], dtype=np.uint8)*255
        for tar_fn in tar_fn_list:
            tar_list = []
            if os.path.isfile(txtfileloc):
                f =  open(txtfileloc, 'r+')
                tar_name = os.path.splitext(os.path.basename(tar_fn))[0]
                textlines = txtremove(f,[tar_name])
                f.seek(0)
                f.write(textlines)
                f.truncate()
                f.close() 
            tar_gray = cv2.imread(tar_fn, cv2.IMREAD_GRAYSCALE)
            #tar_binary = cv2.threshold(tar_gray, 128, 255, cv2.THRESH_BINARY)[1]
            tar_list.append(tar_gray )
            tar_new = cv2.flip(tar_gray ,1)
            tar_list.append(tar_new)
            tar_new = cv2.transpose(tar_gray)
            tar_list.append(tar_new)
            tar_new = cv2.flip(tar_new,-1)
            tar_list.append(tar_new)
            for tar in tar_list:
                # loop over the scales of the image
                w, h = tar.shape[::-1]
                found =  None
                for scale in np.linspace(0.2, 1.0, 20)[::-1]:
		            # resize the image according to the scale, and keep track
		            # of the ratio of the resizing
                    resized = imutils.resize(img_gray, width = int(img_gray.shape[1] * scale))
                    r = img_gray.shape[1] / float(resized.shape[1])

		            # if the resized image is smaller than the template, then break
		            # from the loop
                    if resized.shape[0] < h or resized.shape[1] < w:
                        break

                    # detect edges in the resized, grayscale image and apply template
		            # matching to find the template in the image
                    res = cv2.matchTemplate(resized, tar, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= threshold)
                    if loc[0].size != 0:
                        for pt in zip(*loc[::-1]):
                            (startX, startY) = (int(pt[0] * r), int(pt[1] * r))
                            (endX, endY) = (int((pt[0] + w) * r), int((pt[1] + h) * r))
                            img_crop = img_blank[startY:endY,startX:endX].copy()
                            hc, wc = img_crop.shape[:2]
                            countzero = hc*wc - cv2.countNonZero(img_crop)
                            if countzero *1.0 / (hc*wc) <= 0.7:
                                # if new area is less than 70% of overlap with other proposed masked area
                                cv2.rectangle(img_blank,(startX, startY), (endX, endY), (0,0,0), -1)

                                cv2.rectangle(img_gray,(startX, startY), (endX, endY), (0,0,255), 2)

                                with open(txtfileloc, "a") as f:
                                    txtline = tar_name
                                    txtline += "\t" + str(startX) + "\t" + str(startY) + "\t" + str(endX) + "\t" 
                                    txtline += str(startY) + "\t" + str(endX)+ "\t" + str(endY)+ "\t" + str(startX)+ "\t" + str(endY)
                                    txtline += "\n"
                                    f.writelines(txtline)
                                    f.close()
        cv2.imwrite(os.path.join(output_path, os.path.basename(img_fn)), img_gray)
        

"""
	            # unpack the bookkeeping variable and compute the (x, y) coordinates
	            # of the bounding box based on the resized ratio
	            (_, maxLoc, r) = found
	            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	            (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

                            w, h = tar.shape[::-1]
                            res = cv2.matchTemplate(img_gray,tar,cv2.TM_CCOEFF_NORMED)
                            loc = np.where( res >= threshold)
                            for pt in zip(*loc[::-1]):
                                cv2.rectangle(img_gray, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
                    plt.imshow(img_gray, 'gray'),plt.show()




cwd = os.getcwd()
print(cwd)




img2 = cv2.imread('simple-3.png', 0)
img1 = cv2.imread('reducer_example.jpg', 0)

#img1 = cv2.transpose(img1)
img1 = cv2.flip(img1,1)



#print(np.shape(img2))

#img_gray = cv2.cvtColor(img2 ,  cv2.COLOR_BGR2GRAY)
img_gray = img2.copy()


w, h = img1.shape[::-1]

res = cv2.matchTemplate(img_gray,img1,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img2, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

plt.imshow(img2, 'gray'),plt.show()






MIN_MATCH_COUNT = 3
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
print(np.shape(des1))
print(np.shape(des2))

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)


# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
print(good)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    print(dst)
    img3 = cv2.polylines(img2.copy(),[np.int32(dst)],True,100,3, cv2.LINE_AA)

else:
    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img3,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()




# Initiate SIFT detector
orb = cv2.ORB_create(nfeatures=200)

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

print(len(kp1))

# create BFMatcher object
bf = cv2.BFMatcher( cv2.NORM_L2, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img4  = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img4),plt.show()







# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3,),plt.show()


"""