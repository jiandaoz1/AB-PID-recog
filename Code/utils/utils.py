import cv2
import numpy as np
def txtremove (file,comptypes):
    """
    remove the text line with the same component type 
    """
    lines = ""
    for line in file:
        cnt = [i for i in line.split()]
        if cnt[0] not in set(comptypes):
            lines += line
    return lines


def scale_contour(cnt, scale):
    
    """
    re-scale the contour
    """
    cx, cy = find_center(cnt)

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

def find_center(cnt):
    """
    find contour center
    """
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    return cx, cy