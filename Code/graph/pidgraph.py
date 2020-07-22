import os
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from utils.utils import find_center
import cv2

def create_graph(img_fn_list, shapeloc_path, comploc_path,textloc_path,lineloc_path):

    for img_fn in img_fn_list:
        print(img_fn)
        CompLoc = load_vertices(img_fn, comploc_path)
        ShapeLoc = load_vertices(img_fn, shapeloc_path)
        LN = load_vertices(img_fn, lineloc_path)
        TXTLoc = load_vertices(img_fn, textloc_path)
        LNLoc = [iln for iln in LN if iln[0] == 'LN']
        LNIntersect = [iln for iln in LN if iln[0] != 'LN']
        VertLoc = LNLoc + CompLoc + ShapeLoc
        Vert = {i: VertLoc[i] for i in range(len(VertLoc))} 
        Edges = create_edge(Vert)

        G = nx.Graph()

        for iedge in Edges:
            G.add_edge(iedge[0],iedge[1])
        print("print Vert")
        print(Vert)
        print("print Edges")
        print(Edges)
        LNRemove = []
        Nodes_to_Contract = [[iln[3], iln[4]] for iln in LNIntersect if iln[0] == "CT"]
        for iedge in Nodes_to_Contract:
            G.add_edge(iedge[0],iedge[1])
        Nodes_to_Contract = sorted(Nodes_to_Contract, key=lambda element: (element[1], element[0]),reverse=True)

        Nodels = []
        for inode in Nodes_to_Contract:
            Nodels += inode
        Nodels = set(Nodels)
        nodesgp_to_contract = []
        for inode in Nodels:
            nodegp_to_contract = [inode]
            for jnodes in Nodes_to_Contract:
                if inode == jnodes[0]:
                    nodegp_to_contract.append(jnodes[1])
                elif  inode == jnodes[1]:
                    nodegp_to_contract.append(jnodes[0])
            nodegp_to_contract = np.sort(nodegp_to_contract).tolist()
            nodesgp_to_contract.append(nodegp_to_contract) 
        
        Nodes_to_Contract = np.unique(nodesgp_to_contract)
        for inodes in Nodes_to_Contract: 
            for inode in inodes[1:]:
                if inodes[0] in G.nodes() and inode in G.nodes():
                    G = nx.contracted_nodes(G, inodes[0], inode)

        labeldict = {}
        for key in G.nodes():
            TXTRecog = find_txt(Vert[key], TXTLoc)
            labeldict[key] = str(key)+'-'+Vert[key][0] 
            if TXTRecog != "":
                labeldict[key] += '-' + TXTRecog

        pos = nx.spring_layout(G)
        nx.draw(G,pos, labels=labeldict, with_labels = True)
        plt.show()





def load_vertices(img_fn, input_path):
    LOC = []
    with open(os.path.join (input_path, os.path.splitext(os.path.basename(img_fn))[0]) + "_loc.txt", 'r') as file1:
        for line in file1:
            cnt = [i for i in line.split("\t")]
            loc =[cnt[0]] + [int(i) for i in cnt[1:9]] 
            if len(cnt) > 9:
                loc += cnt[9:]
            LOC.append(loc)
    return LOC

def create_edge2(Vert):
    """
    Find Edges for all vertices
    """
    Edges = []
    for key1 in Vert:
        if Vert[key1][0] == 'LN':
            lnpt1 = Vert[key1][1:3]
            lnpt2 = Vert[key1][3:5]
            DIST1 = [0, 0, 9999.0]
            DIST2 = [0, 0, 9999.0]
            for key2 in Vert:
                if Vert[key2][0] != 'LN':
                    dist1 = find_dist (lnpt1,Vert[key2])
                    dist2 = find_dist (lnpt2,Vert[key2])
                    if dist1 <= DIST1[2]:
                        DIST1 = [key1, key2, dist1]
                    if dist2 <= DIST2[2]:
                        DIST2 = [key1, key2, dist2]
            Edges.append(DIST1)
            Edges.append(DIST2)
    return Edges

def create_edge(Vert):
    """
    Find Edges for all vertices
    """
    Vert_1ln = [i for i in Vert.keys() if Vert[i][0] not in ['VALVE','REDUCER','LN']]
    Vert_2ln = [i for i in Vert.keys() if Vert[i][0] in ['VALVE','REDUCER']]
    Edges = []
    for key1 in Vert.keys():
        if Vert[key1][0] != 'LN':
            DIST = []
            for key2 in Vert:
                if Vert[key2][0] == 'LN':
                    lnpt1 = Vert[key2][1:3]
                    lnpt2 = Vert[key2][3:5]
                    dist1 = find_dist (lnpt1,Vert[key1])
                    dist2 = find_dist (lnpt2,Vert[key1])
                    if dist1 < dist2:
                        DIST.append( [key1, key2, dist1])
                    else:
                        DIST.append( [key1, key2, dist2])
            DIST = sorted(DIST, key=lambda x: x[2])
            if key1 in Vert_1ln:
                Edges.append(DIST[0])
            else:
                Edges.append(DIST[0])
                Edges.append(DIST[1])
    return Edges

def find_dist(pt,Comp):
    """
    find distance between point to a Component Type of Vertices
    """
    cnt = Comp[1:]
    cnt = np.array(cnt).reshape((-1,1,2))
    cx, cy = find_center(cnt)
    dist = ((cx-pt[0])**2 + (cy-pt[1])**2)**0.5
    for i in range(len(cnt)):
        ivt = cnt[i]
        j = i + 1
        if i+1 >= len(cnt):
            j = 0
        jvt = cnt[j]
        cvt = [int((cnt[i][0][0] + cnt[j][0][0])/2) ,int((cnt[i][0][1] + cnt[j][0][1])/2) ]
        if ((cvt[1]-pt[1]) **2 + (cvt[0]-pt[0]) **2)**0.5 < 20 :
            dist = 0
            break
    return dist
    
def find_txt(Comp, txtlist):
    """
    find text for a particular component
    """
    if Comp[0] != "LN":
        cnt_comp = Comp[1:]
        cnt = np.array(cnt_comp).reshape((-1,1,2))
        cx_tar, cy_tar = find_center(cnt)
    else:
        cnt_comp = Comp[1:5]
        cx_tar = int((cnt_comp[0]+cnt_comp[2])/2)
        cy_tar = int((cnt_comp[1]+cnt_comp[3])/2)
    mindist = [9999, 9999,9999]
    for i in range(len(txtlist)):
        cnt_txt = txtlist[i][1:9]
        txtrecog = txtlist[i][9]
        print(txtrecog)
        cnt = np.array(cnt_txt).reshape((-1,1,2))
        cx_txt, cy_txt = find_center(cnt)
        dist = ((cx_tar-cx_txt)**2 + (cy_tar-cy_txt)**2)**0.5
        if dist < mindist[0]:
            mindist = [dist,i,txtrecog]
    if mindist[0] > 50:
        return ""
    else:
        return mindist[2]
    
    
