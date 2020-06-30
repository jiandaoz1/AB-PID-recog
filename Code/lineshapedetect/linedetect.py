
import cv2
import numpy as np
from preprocess.preprocess import get_images
import os
import shutil
from skimage.transform import probabilistic_hough_line
import math
from sklearn.cluster import KMeans
from preprocess.preprocess import remove_border, skeleton,thinning


def houghline_detect(input_path,output_path, lineloc_output_path):
    """
    Houghline detection
    """
    print("========== detect line ==============")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    im_fn_list = get_images(input_path)
    for im_fn in im_fn_list:
        print('===============')
        print(im_fn)
        #edge detection of the line
        im_raw = cv2.imread(im_fn)
        im_draw = im_raw.copy()
        im_gray = cv2.cvtColor(im_raw,cv2.COLOR_BGR2GRAY)

        #im_gray = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY)[1]
        im_gray = cv2.Canny(im_gray,50,100)

        minLineLength = 5
        maxLineGap = 0
        lines = probabilistic_hough_line(im_gray, threshold=10, line_length=minLineLength,line_gap=maxLineGap)
        #lines = cv2.HoughLinesP(im_gray,1,1/180,40,minLineLength,maxLineGap)
        # draw direct output from houghlines

        im_draw2 = im_raw.copy()
        # draw lines and vertice points before merge
        for iln in range(len(lines)):
            lines[iln] = [[lines[iln][0][0],lines[iln][0][1],lines[iln][1][0],lines[iln][1][1]]]
            #for x1, y1, x2, y2 in lines[iln]:
                #cv2.drawMarker(im_draw2, (x1, y1),(0,255,0), markerType=cv2.MARKER_STAR, 
                #                markerSize=4, thickness=1, line_type=cv2.LINE_AA)
                #cv2.drawMarker(im_draw2, (x2, y2),(255,0,0), markerType=cv2.MARKER_STAR, 
                #                markerSize=4, thickness=1, line_type=cv2.LINE_AA)
                #cv2.line(im_draw2,(x1,y1),(x2,y2),(0,0,255),1)

        # Merge the hough lines
        hb = HoughBundler()
        lines = hb.process_lines(lines,im_gray)
        lines = line_connect(lines)
        # draw line ID before merge
        for iln in range(len(lines)):
            lines[iln] = [lines[iln][0][0],lines[iln][0][1],lines[iln][1][0],lines[iln][1][1]]
            x1, y1, x2, y2 = lines[iln]
            cv2.putText(im_draw2, str(iln), (int((x1+x2)/2),int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX ,0.25, (0,255,0), 1, cv2.LINE_AA) 
        cv2.imwrite(os.path.join(output_path, "hough_"+os.path.basename(im_fn)),im_draw2)

        pt_intersect = line_intersect(lines, im_raw)

        for ipt in pt_intersect:
            cv2.drawMarker(im_draw, (ipt[0], ipt[1]),(0,255,0), markerType=cv2.MARKER_STAR, 
                                markerSize=10, thickness=2, line_type=cv2.LINE_AA)
        # draw line ID after merge
        for iln in range(len(lines)):
            p1 = lines[iln][:2]
            p2 = lines[iln][2:]
            cv2.drawMarker(im_draw, (p1[0], p1[1]),(0,255,0), markerType=cv2.MARKER_STAR, 
                                markerSize=4, thickness=1, line_type=cv2.LINE_AA)
            cv2.drawMarker(im_draw, (p2[0], p2[1]),(255,0,0), markerType=cv2.MARKER_STAR, 
                                markerSize=4, thickness=1, line_type=cv2.LINE_AA)
            cv2.line(im_draw,(p1[0],p1[1]),(p2[0],p2[1]),(0,0,255),1)
            with open(os.path.join (lineloc_output_path, os.path.splitext(os.path.basename(im_fn))[0]) + "_lineloc.txt", "a") as f:
                txtline = "LN"
                txtline += "\t" + str(p1[0]) + "\t" + str(p1[1]) + "\t" + str(p2[0]) + "\t" + str(p2[1])
                txtline += "\n"
                f.writelines(txtline)
                f.close()  
        cv2.imwrite(os.path.join(output_path, os.path.basename(im_fn)),im_draw)

def line_connect(lines):
    # setup the max_line_gap to connect the lines aligned
    max_line_gap = 20
    nl = len(lines)
    connect_line = []
    lines_new = []
    for il in range(nl-1):
        for jl in range(il+1, nl):
            iln = lines[il]
            jln = lines[jl]
            ilnp1 = iln[0]
            ilnp2 = iln[1]
            jlnp1 = jln[0]
            jlnp2 = jln[1]
            ln_xm = int(np.mean([ilnp1[0],ilnp2[0],jlnp1[0],jlnp2[0]]))
            ln_ym = int(np.mean([ilnp1[1],ilnp2[1],jlnp1[1],jlnp2[1]]))
            if (abs(ilnp1[0]-ln_xm) < 4 and abs(ilnp2[0]-ln_xm)<4 and abs(jlnp1[0]-ln_xm)<4 and abs(jlnp2[0]-ln_xm) <4):
                # lines run vertically and likely aligned
                ypts = np.sort([ilnp1[1],ilnp2[1],jlnp1[1],jlnp2[1]])
                # calculate two middle point distance
                midis_ypts = abs(ypts[2]-ypts[1])
                if midis_ypts < max_line_gap:
                    connect_line.append([il,jl,[np.array([ln_xm,ypts[0]]),  np.array([ln_xm,ypts[3]])]])

            if (abs(ilnp1[1]-ln_ym) < 4 and abs(ilnp2[1]-ln_ym)<4 and abs(jlnp1[1]-ln_ym)<4 and abs(jlnp2[1]-ln_ym) <4):
                # lines run horizontally and likely aligned
                xpts = np.sort([ilnp1[0],ilnp2[0],jlnp1[0],jlnp2[0]])
                # calculate two middle point distance
                midis_xpts = abs(xpts[2]-xpts[1])
                if midis_xpts < max_line_gap:
                    connect_line.append([il,jl,[np.array([xpts[0],ln_ym]),  np.array([xpts[3],ln_ym])]])
    rm_line_ind = []
    for iln in connect_line:
        rm_line_ind.append(iln[0])
        rm_line_ind.append(iln[1])
        lines_new.append(iln[2])
    rm_line_ind = set(rm_line_ind)
  
    for il in range(nl):
        if il not in rm_line_ind:
            lines_new.append(lines[il])
    return lines_new

def ccw(A,B,C):
    # check for intersection for three points
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def line_intersect(lines,im_raw):
    """
    check for line segment intersect
    """
    nl = len(lines)
    pt_intersect = []
    for il in range(nl-1):
        for jl in range(il+1, nl):
            iln = lines[il]
            jln = lines[jl]
            # calculate min distance between two lines
            hb = HoughBundler()
            min_lndist = hb.get_distance(iln,jln)
            # calculate intersect angle between two lines
            A1 = iln[3]-iln[1]
            B1 = -(iln[2]-iln[0])
            A2 = jln[3]-jln[1]
            B2 = -(jln[2]-jln[0])
            theta = abs(np.arccos((A1*A2+B1*B2)/np.sqrt((A1**2+B1**2)*(A2**2+B2**2))))
            if  (ccw(iln[:2],jln[:2],jln[2:]) != ccw(iln[2:],jln[:2],jln[2:]) and ccw(iln[:2],iln[2:],jln[:2]) != ccw(iln[:2],iln[2:],jln[2:]))\
                or (min_lndist<3 and theta > 70/180*np.pi and theta < 110/180*np.pi):
                # Return true if line segments AB and CD intersect
                xdiff = (iln[0] - iln[2], jln[0] - jln[2])
                ydiff = (iln[1] - iln[3], jln[1] - jln[3])

                def det(a, b):
                    return a[0] * b[1] - a[1] * b[0]

                div = det(xdiff, ydiff)
                if div == 0:
                    print(iln)
                    print(jln)
                    raise Exception('lines do not intersect')
                d = (det(iln[:2],iln[2:]), det(jln[:2],jln[2:]))
                x = int(det(d, xdiff) / div)
                y = int(det(d, ydiff) / div)

                intersect_type = line_intersecttype(iln,jln, [x, y],im_raw)
                pt_intersect.append([x,y,il,jl, intersect_type ])
    return pt_intersect

def line_intersecttype(iln, jln,pt_intersect,img_raw):
    """
    identify line intersection type
    """
    # check intersect within the range of max_line_gap
    max_line_gap = 10
    x0 = pt_intersect[0]
    y0 = pt_intersect[1]
    img_gray = cv2.cvtColor(img_raw,cv2.COLOR_BGR2GRAY)
    # extract pixels around intersection
    img_crop = img_gray[y0-max_line_gap:y0+max_line_gap,x0-max_line_gap:x0+max_line_gap].copy()

    not_cross_pattern = [0.1,0,1,0,0.1]
    print("====================")
    patmatch_x = check_cross_pattern (img_crop,not_cross_pattern)
    img_crop_rot = cv2.transpose(img_crop)
    img_crop_rot = cv2.flip(img_crop_rot,1)
    patmatch_y = check_cross_pattern (img_crop_rot,not_cross_pattern)
    if patmatch_x or patmatch_y:
        # if match, not cross patttern
        ## following pattern is identified
        ###  find a line pattern of :
        ##  255   255  255  0   255  255  255
        ##  255   255  255  0   255  255  255
        ##  255   255  255 255  255  255  255
        ##  0     0    0    0    0    0    0
        ##  255   255  255 255  255  255  255
        ##  255   255  255  0   255  255  255
        ##  255   255  255  0   255  255  255 
        return "CF"
    else:
        return "CT"
    #cv2.imshow('image',img_crop)
    #cv2.waitKey(0)

def check_cross_pattern(img, pattern_target):
    # check if two lines are connected or not at intersection
    h, w = np.shape(img)
    len_target = len(pattern_target)
    pats = []
    for i in range(h):
        ratio = sum(img[i,:] <= 50)/w
        if ratio <0.25 and ratio >0:
            pat = 0.1
        elif ratio ==0:
            pat = 0
        elif ratio >= 0.25 and ratio <= 0.9:
            pat = 0.5
        else:
            pat = 1
        if len(pats)==0 or pats[-1] != pat:
            # add pattern indicator based on the latest number or if there is no number
            pats.append(pat)
    print(pats)
    if len(pats) >= len_target:
        for i in range(len(pats[:-len_target+1])):
            if pats[i:i+len_target] == pattern_target:
                return True
            else:
                return False


    

class HoughBundler:
    '''Clasterize and merge each cluster of cv2.HoughLinesP() output
    a = HoughBundler()
    foo = a.process_lines(houghP_lines, binary_image)
    '''

    def get_orientation(self, line):
        '''get orientation of a line, using its length
        https://en.wikipedia.org/wiki/Atan2
        '''
        orientation = math.atan2(abs((line[0] - line[2])), abs((line[1] - line[3])))
        return math.degrees(orientation)

    def checker(self, line_new, groups, min_distance_to_merge, min_angle_to_merge):
        '''Check if line have enough distance and angle to be count as similar
        '''
        for group in groups:
            # walk through existing line groups
            for line_old in group:
                # check distance
                if self.get_distance(line_old, line_new) < min_distance_to_merge:
                    # check the angle between lines
                    orientation_new = self.get_orientation(line_new)
                    orientation_old = self.get_orientation(line_old)
                    # if all is ok -- line is similar to others in group
                    if abs(orientation_new - orientation_old) < min_angle_to_merge:
                        group.append(line_new)
                        return False
        # if it is totally different line
        return True

    def DistancePointLine(self, point, line):
        """Get distance between point and line
        http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        """
        px, py = point
        x1, y1, x2, y2 = line

        def lineMagnitude(x1, y1, x2, y2):
            'Get line (aka vector) length'
            lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return lineMagnitude

        LineMag = lineMagnitude(x1, y1, x2, y2)
        if LineMag < 0.00000001:
            DistancePointLine = 9999
            return DistancePointLine

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (LineMag * LineMag)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = lineMagnitude(px, py, x1, y1)
            iy = lineMagnitude(px, py, x2, y2)
            if ix > iy:
                DistancePointLine = iy
            else:
                DistancePointLine = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            DistancePointLine = lineMagnitude(px, py, ix, iy)

        return DistancePointLine


    def merge_lines_pipeline_2(self, lines):
        'Clusterize (group) lines'
        groups = []  # all lines groups are here
        # Parameters to play with
        min_distance_to_merge = 3
        min_angle_to_merge = 20
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.checker(line_new, groups, min_distance_to_merge, min_angle_to_merge):
                groups.append([line_new])
        return groups

    def merge_lines_segments1(self, lines):
        """Sort lines cluster and return first and last coordinates
        """
        orientation = self.get_orientation(lines[0])

        # special case
        if(len(lines) == 1):
            return [lines[0][:2], lines[0][2:]]

        # [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        # if vertical
        if 45 < orientation < 135:
            #sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            #sort by x
            points = sorted(points, key=lambda point: point[0])

        # return first and last point in sorted group
        # [[x,y],[x,y]]
        return [points[0], points[-1]]

    def process_lines(self, lines, img):
        '''Main function for lines from cv.HoughLinesP() output merging
        for OpenCV 3
        lines -- cv.HoughLinesP() output
        img -- binary image
        '''
        lines_x = []
        lines_y = []
        intersept_y = []
        intersept_x = []
        # for every line of cv2.HoughLinesP()
        
        for line_i in [l[0] for l in lines]:
                orientation = self.get_orientation(line_i)
                # if vertical
                if 45 < orientation < 135:
                    lines_y.append(line_i)
                    intersept_y.append(-line_i[2]*(line_i[3]-line_i[1])/(line_i[2]-line_i[0])+line_i[3])

                else:
                    lines_x.append(line_i)
                    intersept_x.append(-line_i[3]*(line_i[2]-line_i[0])/(line_i[3]-line_i[1])+line_i[2])


        num_of_cluster = 1
        lines_yy =[]
        lines_xx = []
        if len(lines_y) > 0:
            kmeans_y = KMeans(n_clusters=num_of_cluster, random_state=0).fit(np.array(intersept_y).reshape(-1,1))
            for ilabel in range(num_of_cluster):
                lines = np.array(lines_y)[kmeans_y.labels_==ilabel]
                if len(lines) > 1:
                    lines = sorted(lines, key=lambda line: line[1])
                lines_yy.append(lines)
        if len(lines_x) > 0 :
            kmeans_x = KMeans(n_clusters=num_of_cluster, random_state=0).fit(np.array(intersept_x).reshape(-1,1))
            for ilabel in range(num_of_cluster):
                lines = np.array(lines_x)[kmeans_x.labels_==ilabel]
                if len(lines) > 1:
                   lines = sorted(lines, key=lambda line: line[0])
                lines_xx.append(lines)

        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_xx, lines_yy]:
            for j in i:
                if len(j) > 0:
                    groups = self.merge_lines_pipeline_2(j)
                    merged_lines = []
                    for group in groups:
                        merged_lines.append(self.merge_lines_segments1(group))

                    merged_lines_all.extend(merged_lines)

        return merged_lines_all

    def get_distance(self, a_line, b_line):
        """Get all possible distances between each dot of two lines and second line
        return the shortest
        """

        dist1 = self.DistancePointLine(a_line[:2], b_line)
        dist2 = self.DistancePointLine(a_line[2:], b_line)
        dist3 = self.DistancePointLine(b_line[:2], a_line)
        dist4 = self.DistancePointLine(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)