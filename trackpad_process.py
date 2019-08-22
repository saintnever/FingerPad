import serial
import matplotlib
matplotlib.use('TkAgg') # MUST BE CALLED BEFORE IMPORTING plt
from matplotlib import pyplot as plt
import queue
import threading
from matplotlib import animation
import seaborn as sns
import numpy as np
import time
#import MLX90640
import cv2 as cv
from collections import deque
import math
import colorsys
from skimage import morphology, data
from skimage.util import invert
from scipy import ndimage, spatial
from mahotas.morph import hitmiss
import os
import pickle

class SerialReader(threading.Thread):
    def __init__(self, stop_event, sig, serport):
        threading.Thread.__init__(self)
        self.stopped = stop_event
        self.signal = sig
        self.n = 1538
        self.frame = 0
        self.temp = [0] * int(self.n / 2)
        port = serport
        # self.s = serial.Serial(port, 9600, timeout=1, rtscts=True, dsrdtr=True)
        self.s = serial.Serial(port, 460800, timeout=0.1, rtscts=False, dsrdtr=False)
        if not self.s.isOpen():
            self.s.open()
        print("connected: ", self.s)
        # self.s.setDTR(True)

    def run(self):
        cnt = 0
        while not self.stopped.is_set():
            try:
                d = ord(self.s.read())
                if d == 0x5a:
                    d = ord(self.s.read())
                    if d == 0x5a:
                        # get the frame length
                        n = self.s.read(2)
                        # read frame
                        self.frame = self.s.read(self.n)
                        # calculate and compare CRC
                        crc_read = self.s.read(2)
                        crc_r = ((crc_read[1] << 8) & 0xFF00) + crc_read[0]
                        crc_cal = self.n_CRC(self.frame, self.n)
                        if crc_r == crc_cal:
                            self.signal.put([time.time(),self.temp[:-1]])
                            cnt += 1
                        else:
                            print('Bad Frame Detected!')
                            print('read crc is {0:x}, cal crc is {1:x}'.format(crc_r, crc_cal))
            except:
                continue
        self.clean()

    def n_CRC(self, data, n):
        crc_cal = 23130 + n
        i = 0
        while i < n:
            j = ((data[i+1] << 8) & 0xFF00) + data[i]
            self.temp[int(i / 2)] = j/100.0
            crc_cal = j + crc_cal
            i = i + 2
        return crc_cal & ((1 << 16) - 1)

    def get_signal(self):
        return self.signal
        
    def clean(self):
        # self.s.cancel_read()
        while self.s.isOpen():
            self.s.close()
            # print('the serial port is open? {}'.format(self.s.isOpen()))


def colorscale(datas, minc, maxc):
    data_scales = list()
    # maxc = np.max(datas)
    # minc = np.min(datas)
    for data in datas:
        data_scale = 256 * (data - minc) / (maxc - minc)
        if data_scale < 0.5*256:
        # if data_scale < 0:
            data_scale = 0
        elif data_scale > 255:
            data_scale = 255
        data_scales.append(data_scale)
    # print(data, data_scale)
    return data_scales

# def colorscale(mat):
#     mmax = np.amax(mat)
#     mmin = np.amin(mat)
#     smat = np.array(256 * (mat-mmin) / (mmax-mmin), np.uint8)
#     smat[smat < 0.5 * 256] = 0
#     smat[smat > 255] = 255
#     return smat

def find_furthest(contour, center):
    dmax = 0
    index = 0
    for item in contour:
        x = item[0][0]
        y = item[0][1]
        # if x == 0 or x == 31 or y == 0 or y == 23:
        #     continue
        d = (x-center[0])**2 + (y-center[1])**2
        if d > dmax and x * y != 0 and x != re_size[0] - 1 and y != re_size[1] - 1:
        # if d > dmax:
            dmax = d
            index = [x, y]
    return index

def trackFingertip(img, tip_prev):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # cv2.RETR_TREE
    areas = [cv.contourArea(c) for c in contours]
    if areas:
        # print(areas)
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        x, y, w, h = cv.boundingRect(cnt)
        center = [x + w / 2, y + h / 2]
        ftip = find_furthest(contours[max_index], center)
        # dist = math.sqrt((ftip[0] - tip_prev[0]) ** 2 + (ftip[1] - tip_prev[1]) ** 2)
        # print(dist)
        # if dist > 20 and (tip_prev[0] != -1 and tip_prev[1] != -1):
        #     return tip_prev
    else:
        ftip = None
    return ftip

def map_def(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
def constrain(val, min_val, max_val):
    max_array = np.empty(val.shape)
    max_array.fill(max_val)
    min_array = np.empty(val.shape)
    min_array.fill(min_val)
    return np.minimum(max_array, np.maximum(min_array, val))

def image_filter(img):
    fimg = np.array(img, np.float)
    t_max = np.amax(fimg)
    t_min = np.amin(fimg)
    index_blue = (fimg <= (t_min+1))
    index = (fimg > (t_min+1))
    fimg[index] = constrain(map_def(fimg[index], t_min, t_max, 0.5, 1.0), 0.5, 1.0)
    fimg[fimg < 0] = 0
    fimg[index_blue] = 0.667
    rgbimg = np.zeros((img0.shape[0], img0.shape[1], 3), np.float32)
    for i in range(fimg.shape[0]):
        for j in range(fimg.shape[1]):
            rgbimg[i][j] = colorsys.hsv_to_rgb(fimg[i][j], 1.0, 1.0)
    return rgbimg

def onedge(point, fsize, dis):
    for i in range(len(point)):
        if abs(point[i] - fsize[i]) < dis or point[i] < dis:
            return True
    return False

def wristedge(img, wedge_prev):  # img need to be binary image
    ratios = [0] * 8
    edge_top = img[0]
    edge_bottom = img[-1]
    edge_left = np.array([item[0] for item in img])
    edge_right = np.array([item[-1] for item in img])
    width = len(edge_left)
    height = len(edge_top)
    # print(width, height)
    qcircum = (width+height)/2
    # ensure counter-clockwise sequence
    ratios[0] = np.sum(edge_top)/width
    ratios[1] = np.sum(edge_left)/height
    ratios[2] = np.sum(edge_bottom)/width
    ratios[3] = np.sum(edge_right)/height
    ratios[4] = (np.sum(edge_top[0:int(width / 2)]) + np.sum(edge_left[0:int(height / 2)])) / qcircum
    ratios[5] = (np.sum(edge_left[-int(height / 2):]) + np.sum(edge_bottom[0:int(width / 2)])) / qcircum
    ratios[6] = (np.sum(edge_bottom[-int(width / 2):]) + np.sum(edge_right[-int(height / 2):])) / qcircum
    ratios[7] = (np.sum(edge_right[0:int(height / 2)]) + np.sum(edge_top[-int(width / 2):])) / qcircum
    # ratios[5] = (np.sum(imgT[0][0:int(fsy / 2)]) + np.sum(img[-1][0:int(fsx / 2)])) / fsd
    # ratios[6] = (np.sum(img[-1][0:int(fsx / 2)]) + np.sum(imgT[-1][0:int(fsy / 2)])) / fsd
    # ratios[7] = (np.sum(imgT[-1][0:int(fsy / 2)]) + np.sum(img[0][0:int(fsx / 2)])) / fsd
    # ratios_sorted = np.sort(ratios)
    # for i, item in enumerate(ratios):
    #     if i == len(ratios)-1:
    #         if ratios[i] > 0.4 and ratios[0] > 0.4:
    #             return i+4
    #     else:
    #         if ratios[i] > 0.4 and ratios[i+1] > 0.4:
    #             return i+4

    if np.sum(ratios) < 1:
        return ratios.index(max(ratios))
    else:
        return wedge_prev

def find_fingertip(wedge, ep_ind, framesize, indextip_prev=None):
    dmax = 0
    indextip = indextip_prev
    for i in range(0,len(ep_ind[0])):
        epind = [ep_ind[0][i], ep_ind[1][i]]
        # jump = math.sqrt((epind[0]-indextip_prev[0])**2 +(epind[1]-indextip_prev[1])**2)
        # if jump > np.max(framesize)*0.3:
        #     continue
        # epind = [int(framesize[0]/2)-10, int(framesize[1]/2)-10]
        dist = 0
        if wedge == 0:
            dist = epind[0]
        elif wedge == 1:
            dist = epind[1]
        elif wedge == 2:
            dist = framesize[0] - epind[0]
        elif wedge == 3:
            dist = framesize[1] - epind[1]
        elif wedge == 4:
            dist = math.sqrt(epind[0]**2 + epind[1]**2)
        elif wedge == 5:
            dist = math.sqrt(epind[1]**2 + (framesize[0]-epind[0])**2)
        elif wedge == 6:
            dist = math.sqrt((framesize[0]-epind[0])**2 + (framesize[1]-epind[1])**2)
        elif wedge == 7:
            dist = math.sqrt((framesize[1]-epind[1])**2 + epind[0]**2)
        # print(wedge, dist)
        if dist > dmax:
            dmax = dist
            indextip = epind
    return indextip

def branchPoints(skel):
    X=[]
    #cross X
    X0 = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]])
    X1 = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1]])
    X.append(X0)
    X.append(X1)
    #T like
    T=[]
    #T0 contains X0
    T0=np.array([[2, 1, 2],
                 [1, 1, 1],
                 [2, 2, 2]])

    T1=np.array([[1, 2, 1],
                 [2, 1, 2],
                 [1, 2, 2]])  # contains X1

    T2=np.array([[2, 1, 2],
                 [1, 1, 2],
                 [2, 1, 2]])

    T3=np.array([[1, 2, 2],
                 [2, 1, 2],
                 [1, 2, 1]])

    T4=np.array([[2, 2, 2],
                 [1, 1, 1],
                 [2, 1, 2]])

    T5=np.array([[2, 2, 1],
                 [2, 1, 2],
                 [1, 2, 1]])

    T6=np.array([[2, 1, 2],
                 [2, 1, 1],
                 [2, 1, 2]])

    T7=np.array([[1, 2, 1],
                 [2, 1, 2],
                 [2, 2, 1]])
    T.append(T0)
    T.append(T1)
    T.append(T2)
    T.append(T3)
    T.append(T4)
    T.append(T5)
    T.append(T6)
    T.append(T7)
    #Y like
    Y=[]
    Y0=np.array([[1, 0, 1],
                 [0, 1, 0],
                 [2, 1, 2]])

    Y1=np.array([[0, 1, 0],
                 [1, 1, 2],
                 [0, 2, 1]])

    Y2=np.array([[1, 0, 2],
                 [0, 1, 1],
                 [1, 0, 2]])

    Y2=np.array([[1, 0, 2],
                 [0, 1, 1],
                 [1, 0, 2]])

    Y3=np.array([[0, 2, 1],
                 [1, 1, 2],
                 [0, 1, 0]])

    Y4=np.array([[2, 1, 2],
                 [0, 1, 0],
                 [1, 0, 1]])
    Y5=np.rot90(Y3)
    Y6 = np.rot90(Y4)
    Y7 = np.rot90(Y5)
    Y.append(Y0)
    Y.append(Y1)
    Y.append(Y2)
    Y.append(Y3)
    Y.append(Y4)
    Y.append(Y5)
    Y.append(Y6)
    Y.append(Y7)

    bp = np.zeros(skel.shape, dtype=int)
    for x in X:
        bp = bp + hitmiss(skel,x)
    for y in Y:
        bp = bp + hitmiss(skel,y)
    for t in T:
        bp = bp + hitmiss(skel,t)

    return bp

def endPoints(skel):
    endpoint1 = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [2, 1, 2]])

    endpoint2=np.array([[0, 0, 0],
                        [0, 1, 2],
                        [0, 2, 1]])

    endpoint3=np.array([[0, 0, 2],
                        [0, 1, 1],
                        [0, 0, 2]])
    
    endpoint4=np.array([[0, 2, 1],
                        [0, 1, 2],
                        [0, 0, 0]])

    endpoint5=np.array([[2, 1, 2],
                        [0, 1, 0],
                        [0, 0, 0]])

    endpoint6=np.array([[1, 2, 0],
                        [2, 1, 0],
                        [0, 0, 0]])

    endpoint7=np.array([[2, 0, 0],
                        [1, 1, 0],
                        [2, 0, 0]])

    endpoint8=np.array([[0, 0, 0],
                        [2, 1, 0],
                        [1, 2, 0]])

    ep1=hitmiss(skel, endpoint1)
    ep2=hitmiss(skel, endpoint2)
    ep3=hitmiss(skel, endpoint3)
    ep4=hitmiss(skel, endpoint4)
    ep5=hitmiss(skel, endpoint5)
    ep6=hitmiss(skel, endpoint6)
    ep7=hitmiss(skel, endpoint7)
    ep8=hitmiss(skel, endpoint8)
    ep = ep1 + ep2 + ep3 + ep4 + ep5 + ep6 + ep7 + ep8
    # ep = ep1 + ep2 + ep8
    return ep

def find_indextip(bp0_ind, ep0_ind, indextip_prev, bp_prev):
    indextip = indextip_prev
    bp = bp_prev
    # if len(ep0_ind[0]) == 1:
    #     return [ep0_ind[0][0], ep0_ind[1][0]]
    if len(bp0_ind[0]) == 1:
        bp = bp0_ind
        bp_prev = bp0_ind
    if len(ep0_ind[0]) >= 1:
        dmax = 0
        for i in range(len(ep0_ind[0])):
            # the point is on the edge
            # if onedge([ep0_ind[0][i], ep0_ind[1][i]], re_size, 5):
            #     continue
            epind = [ep0_ind[0][i], ep0_ind[1][i]]
            dist = (epind[0] - bp[0][0])**2 + (epind[1] - bp[1][0])**2
            if dist > dmax:
                dmax = dist
                if (epind[1]-bp[1][0]) * (indextip[1]-bp[0][0]) > 0:
                    indextip = epind
            # if (epind[0] - bp[0][0])*dir[0] > 0 and (epind[1] - bp[1][0])*dir[1] > 0:
            #     indextip = [ep0_ind[0][i], ep0_ind[1][i]]
    # initial state

    return indextip, bp_prev

    # dist = math.sqrt((indextip[0]-indextip_prev[0])**2 + (indextip[1]-indextip_prev[1])**2)
    # if dist < 50:
    #     return indextip
    # else:
    #     return indextip_prev

def distance(point1, point2=None):
    if point2 is not None:
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    else:
        return math.sqrt(point1[0] ** 2 + point1[1] ** 2)

def find_nearestn(point, contour, n):
    dists = list()
    for p in contour:
        # print(point, list(p[0]))
        dists.append(distance(point, list(p[0])))
    dists_sort = sorted(dists)
    if len(dists) < n:
        n = len(dists)
    points = list()
    # print(dists_sort)
    for i in range(n):
        points.append(contour[dists.index(dists_sort[i])][0])
    return np.array(points, np.int32)


def getPerpCoord(aX, aY, bX, bY, length):
    vX = bX-aX
    vY = bY-aY
    #print(str(vX)+" "+str(vY))
    if(vX == 0 or vY == 0):
        return 0, 0, 0, 0
    mag = math.sqrt(vX*vX + vY*vY)
    vX = vX / mag
    vY = vY / mag
    temp = vX
    vX = 0-vY
    vY = temp
    pX = aX + (bX-aX) * 0.1
    pY = aY + (bX-aX) * 0.1
    cX = pX + vX * length
    cY = pY + vY * length
    dX = pX - vX * length
    dY = pY - vY * length
    return int(cX), int(cY), int(dX), int(dY)

re_size = (32*5, 32*5)
path = './trackpad_model_data/'

if __name__ == '__main__':
    try:
        xrg = 11
        yrg = 11
        # cimg = [[[] for _ in range(yrg)] for _ in range(xrg)]

        # read and save images
        # filenames = [filename for filename in os.listdir(path) if filename.endswith('jpg')]
        # for file in filenames:
        #     xcur = int(file.split('_')[0])
        #     ycur = int(file.split('_')[1])
        #     img_raw = cv.imread(path + file)
        #     img_raw = cv.cvtColor(img_raw, cv.COLOR_BGR2GRAY)
        #     cimg[xcur][ycur].append(img_raw)
        #
        # with open(path + 'images_model.pkl', 'wb') as file:
        #     pickle.dump(cimg, file)

        with open(path + 'temps_model_hot.pkl', 'rb') as file:
            temps = pickle.load(file)

        # with open(path + 'images_model.pkl', 'rb') as file:
        #     cimg = pickle.load(file)
        ftips = list()
        dist_list = [[0 for _ in range(xrg)] for _ in range(yrg)]
        for j in range(xrg):
            for i in range(yrg):
        # for i in [6]:
        #     for j in [4]:
                # img = np.mean(cimg[i][j], axis=0)
                # img = cimg[i][j][int(len(cimg[i][j]) / 2)]
                temp = np.amin(temps[i][j], axis=0)
                # temp = temps[i][j][int(len(temps[i][j]) / 2)]
                temp_scale = colorscale(temp, np.min(temp), np.max(temp))
                # preprocess
                img_raw = np.array([[0] * 32 for _ in range(24)], np.float)
                img_scale = np.array([[0] * 32 for _ in range(24)], np.float)
                for k, x in enumerate(temp):
                    row = k // 32
                    col = 31 - k % 32
                    img_raw[int(row)][int(col)] = x
                    img_scale[int(row)][int(col)] = temp_scale[k]

                img_raw = cv.flip(img_raw, 0)
                img_raw = cv.resize(img_raw, (len(img_raw[0])*10, len(img_raw)*10), interpolation=cv.INTER_CUBIC)
                blur_raw = cv.GaussianBlur(img_raw, (25, 25), 0)

                img = cv.flip(img_scale, 0)
                img = cv.resize(img, (len(img[0]) * 10, len(img) * 10), interpolation=cv.INTER_CUBIC)
                blur = cv.GaussianBlur(img, (25, 25), 0)

                blur[blur < 0] = 0
                blur_raw[blur_raw < 0] = 0
                blur_bi = blur.astype(np.uint8)
                # otsu binarization
                ret, th0 = cv.threshold(blur_bi, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                # contour mask assuming the largest contour is hand
                contours, hierarchy = cv.findContours(th0, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # cv2.RETR_TREE
                areas = [cv.contourArea(c) for c in contours]
                cnt = []
                mask = np.zeros(th0.shape)
                indextip = (0, 0)
                if areas:
                    max_index = np.argmax(areas)
                    cnt = contours[max_index]
                    x, y, w, h = cv.boundingRect(cnt)
                    cv.rectangle(blur, (x, h), (w, y), (0, 255, 0), 3)

                    hull = cv.convexHull(cnt, returnPoints=False)
                    defects = cv.convexityDefects(cnt, hull)
                    starts = list()
                    ends = list()
                    ymin = 1000
                    xmax = 0
                    distmax = 0
                    p1 = 0
                    p2 = 0
                    for l in range(defects.shape[0]):
                        s, e, f, d = defects[l, 0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        if start[0] > xmax:
                            xmax = start[0]
                            p1 = start
                        if start[1] < ymin:
                            ymin = start[1]
                            p2 = start
                        if end[0] > xmax:
                            xmax = end[0]
                            p1 = end
                        if end[1] < ymin:
                            ymin = end[1]
                            p2 = end
                        # if distance(start) > distmax:
                        #     distmax = distance(start)
                        #     indextip = start
                        # if distance(end) > distmax:
                        #     distmax = distance(end)
                        #     indextip = end

                        # far = tuple(cnt[f][0])
                        # cv.circle(blur, far, 5, (0, 255, 255), -1)

                    # cv.circle(blur, p1, 5, (0, 255, 255), -1)
                    cv.line(blur, p1, p2, [255, 255, 255], 1)
                    c0, c1, d0, d1 = getPerpCoord(p1[0], p1[1], p2[0], p2[1], 50)
                    tmp = np.zeros_like(blur, np.uint8)
                    cv.line(tmp, (c0, c1), (d0, d1), [255, 255, 255], 5)
                    ref = np.zeros_like(blur, np.uint8)
                    ref = cv.drawContours(ref, [cnt], -1, (255, 255, 255), 1)
                    # Step #6d
                    (x_intercept, y_intercept) = np.nonzero(np.logical_and(tmp, ref))
                    p1 = np.array(p1)
                    p2 = np.array(p2)
                    dists = list()
                    for m in range(len(x_intercept)):
                        p3 = np.array([y_intercept[m], x_intercept[m]])
                        cv.circle(blur, tuple(p3), 2, (0, 255, 255), -1)
                        dist = abs(np.cross(p2 - p1, p1 - p3) / np.linalg.norm(p2 - p1))
                        if dist > 8 and dist < 40:
                            dists.append(dist)

                    cv.line(blur, (c0, c1), (d0, d1), [255, 255, 255], 5)
                    # cv.drawContours(blur, cnt[hull], -1, (255, 255, 255), 2)
                    # Create a mask from the largest contour
                    cv.fillPoly(mask, [cnt], 1)
                    # Use mask to crop data from original image
                    th0 = np.multiply(th0, mask)
                    blur0 = np.multiply(blur_raw, mask)
                # skeletonize using Zhang-seun method
                th0 = th0.astype(np.bool)
                th0_s = morphology.skeletonize(th0)
                # calcualte endpoints using hit-or-miss
                ep0 = endPoints(th0_s)
                ep0_ind = np.where(ep0 == 1)
                # fingertip is the endpoint furthest to a corner
                indextip0 = [-1, -1]
                if len(ep0_ind[0]) >= 1:
                    indextip0 = find_fingertip(4, ep0_ind, re_size)

                fit_points = find_nearestn([indextip0[1], indextip0[0]], cnt, 20)
                # for p in fit_points:
                #     cv.circle(blur, tuple(p[0]), 1, (255, 255, 255), -1)
                rrect = cv.fitEllipse(fit_points)
                cv.ellipse(blur, rrect, (255, 255, 255), 1, cv.LINE_AA)
                # cv.drawContours(blur, [fit_points], -1, (255, 255, 255), 2)
                mask_ft = np.zeros(th0.shape)

                mask_ft = np.multiply(mask_ft, mask)

                # th0[indextip0[0]:] = 0
                cv.circle(blur, (indextip0[1], indextip0[0]), 2, (255, 255, 255), -1)
                # area ratio
                ar = np.sum(np.sum(th0)) / (len(th0) * len(th0.transpose()))
                wr = np.sum(th0.transpose()[0]) / len(th0)
                if len(dists) != 0:
                    print('x:{}, y:{}, width:{}, area ratio:{}, wrist ratio:{}'.format(i, j, np.nanmin(dists), ar, wr))
                    dist_list[i][j] = np.nanmin(dists)
                # blur_raw = np.multiply(blur_raw, mask_ft.astype(bool))
                #
                # # cv.fillPoly(mask_ft, [cnt], 1)
                # # plt.figure()
                # # plt.imshow(th0_s, cmap='seismic')
                # # plt.title(str(i)+'_'+str(j))
                # # plt.show()
                #
                # ref = np.zeros_like(th0, np.uint8)
                # ref = cv.drawContours(ref, [cnt], -1, (255, 255, 255), 1)
                # # Define total number of angles we want
                # N = 20
                # dists = list()
                # # Step #6
                # lines = list()
                # for l in range(N):
                #     # Step #6b
                #     theta = l * (360 / N)
                #     theta *= np.pi / 180.0
                #     # Step #6c
                #     tmp = np.zeros_like(th0, np.uint8)
                #
                #     cv.line(tmp, (int(indextip0[1] - np.cos(theta) * 100),
                #               int(indextip0[0] + np.sin(theta) * 100)),
                #              (int(indextip0[1] + np.cos(theta) * 100),
                #               int(indextip0[0] - np.sin(theta) * 100)), 255, 1)
                #
                #     # Step #6d
                #     (y_intercept, x_intercept) = np.nonzero(np.logical_and(tmp, ref))
                #     if len(x_intercept) < 2:
                #         continue
                #     elif len(x_intercept) == 2:
                #         id0 = 0
                #         id1 = 1
                #     else:
                #         id0 = 0
                #         id1 = 0
                #         for m, xp in enumerate(x_intercept):
                #             if xp < indextip0[1]:
                #                 continue
                #             else:
                #                 id0 = m-1
                #                 id1 = m
                #                 break
                #
                #     dist = math.sqrt((x_intercept[id0] - x_intercept[id1]) ** 2 + (y_intercept[id0] - y_intercept[id1]) ** 2)
                #     if dist == 0:
                #         continue
                #     lines.append([(x_intercept[id0], y_intercept[id0]), (x_intercept[id1], y_intercept[id1])])
                #     dists.append(dist)
                # id_width = np.argmin(dists)
                ftips.append([i, j, (indextip[1], indextip[0]), np.mean(blur_raw[blur_raw>0])])
                # ftips.append([i, j, (indextip[1], indextip[0]), ])
                # cv.line(blur, lines[id_width][0], lines[id_width][1], 0, 1)
                # blur = cv.circle(blur, (indextip0[1], indextip0[0]), 5, (0, 127, 255), -1)

                # plt.figure()
                # fig, ([ax0, ax1]) = plt.subplots(1,2)
                # ax0.imshow(blur, cmap='seismic')
                # ax1.imshow(th0, cmap='seismic')
                # plt.title(str(i) + '_' + str(j))
                # plt.show()

                # # Step #6e
                # cv.line(blur, (indextip0[1], indextip0[0]), (x_intercept[-1], y_intercept[-1]), 0, 1)

                # blur = cv.circle(blur, (indextip0[1], indextip0[0]), 5, (0, 127, 255), -1)


                # break

        # for l in range(xrg):
        #     tmp = list()
        #     for item in ftips:
        #         if item[1] == l:
        #             tmp.append(item[-1])
        #     print(tmp)
        # plt.figure()
        # plt.imshow(tmp, cmap='seismic')
        # plt.show()
        #
        # plt.figure()
        # plt.imshow(blur, cmap='seismic')
        # plt.show()
        # fig, (ax0) = plt.subplots(1, 1)
        # im0 = ax0.imshow(np.random.uniform(low=20, high=35, size=re_size), cmap='seismic')
        # plt.tight_layout()
        # plt.ion()
        # while True:
        #
        #     time_loop = round(time.monotonic() - time_start, 2)
        #     tcnt = int(time_loop) // t
        #     xcur = tcnt // xrg + 10
        #     ycur = tcnt % yrg + 9
        #     filename = str(xcur) + '_' + str(ycur) + '_' + str(time_loop)
        #     if int(time_loop) % t < 3:
        #         print('MOVE! x:{}, y:{}'.format(xcur, ycur))
        #     else:
        #         print('STAY! x:{}, y:{}'.format(xcur, ycur))
        #         cv.imwrite(path+filename+'.jpg', img0)
        #     im0.set_array(img0)
        #     plt.pause(0.001)

    finally:
        cv.destroyAllWindows()
