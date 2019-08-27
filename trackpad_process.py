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
import pandas as pd

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

re_size = (24*10, 32*10)
path = './trackpad_model_data/'

if __name__ == '__main__':
    try:
        xrg = 6
        yrg = 6
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

        with open(path + 'temps_model_flat.pkl', 'rb') as file:
            temps = pickle.load(file)

        # with open(path + 'images_model.pkl', 'rb') as file:
        #     cimg = pickle.load(file)
        ftips = list()
        dist_list = [[0 for _ in range(xrg)] for _ in range(yrg)]
        tp_model = list()
        # tp_model = pd.DataFrame(index=np.arange(0, xrg*yrg), columns=['i', 'j', 'xpos', 'ypos', 'h', 'h_rotate', 'dmax', 'finger_width'])
        cnt = 0
        for i in range(xrg):
            for j in range(yrg):
        # for i in [5]:
        #     for j in [0]:
                # img = np.mean(cimg[i][j], axis=0)
                # img = cimg[i][j][int(len(cimg[i][j]) / 2)]
                temp = np.mean(temps[i][j], axis=0)
                # temp = temps[i][j][int(len(temps[i][j]) / 2)]
                # temp = temps[i][j][-2]
                temp_scale = colorscale(temp, np.min(temp), np.max(temp))
                # preprocess
                img_raw = np.array([[0] * 32 for _ in range(24)], np.float)
                img_scale = np.array([[0] * 32 for _ in range(24)], np.float)
                for k, x in enumerate(temp):
                    row = k // 32
                    col = 31 - k % 32
                    img_raw[int(row)][int(col)] = x
                    img_scale[int(row)][int(col)] = temp_scale[k]

                img_raw[img_raw < 130] = 0
                img_raw = cv.flip(img_raw, 0)
                img_raw = cv.resize(img_raw, (len(img_raw[0])*10, len(img_raw)*10), interpolation=cv.INTER_CUBIC)
                blur_raw = cv.GaussianBlur(img_raw, (25, 25), 0)

                img_scale[img_scale < 130] = 0
                img = cv.flip(img_scale, 0)
                img = cv.resize(img, (len(img[0]) * 10, len(img) * 10), interpolation=cv.INTER_CUBIC)
                blur = cv.GaussianBlur(img, (31, 31), 0)

                blur[blur < 0] = 0
                blur_raw[blur_raw < 0] = 0
                blur_bi = blur.astype(np.uint8)
                # otsu binarization
                ret, th0 = cv.threshold(blur_bi, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                kerneld = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
                th0 = cv.dilate(th0, kerneld)

                # contour mask assuming the largest contour is hand
                contours, hierarchy = cv.findContours(th0, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # cv2.RETR_TREE
                areas = [cv.contourArea(c) for c in contours]
                dists = list()

                if len(areas) == 0:
                    continue
                max_index = np.argmax(areas)
                cnt = contours[max_index]
                hull = cv.convexHull(cnt, returnPoints=False)
                x, y, w, h = cv.boundingRect(cnt)
                center = [int(x + w / 2), int(y + h / 2)]
                mask = np.zeros_like(blur)
                xmax = 0
                ymin = 10000
                xmin = 10000
                p0 = 0
                ifp = 0
                ph = 0
                iwb = 0
                temp0 = np.zeros_like(blur)
                for k in range(len(hull)):
                    if cnt[hull[k]][0][0][0] > xmax:
                        xmax = cnt[hull[k]][0][0][0]
                        p0 = tuple(cnt[hull[k]][0][0])
                        ifp = k
                    if cnt[hull[k]][0][0][1] < ymin:
                        ymin = cnt[hull[k]][0][0][1]
                        ph = tuple(cnt[hull[k]][0][0])
                    if cnt[hull[k]][0][0][0] <= xmin:
                        if cnt[hull[k]][0][0][0] == xmin:
                            if cnt[hull[k]][0][0][1] > cnt[hull[iwb]][0][0][1]:
                                iwb = k
                        xmin = cnt[hull[k]][0][0][0]
                        iwb = k - 1
                    if k < len(hull) - 1:
                        cv.line(temp0, tuple(cnt[hull[k]][0][0]), tuple(cnt[hull[k + 1]][0][0]), [127, 255, 255], 3)
                    else:
                        cv.line(temp0, tuple(cnt[hull[k]][0][0]), tuple(cnt[hull[0]][0][0]), [127, 255, 255], 3)
                pb = (0, 0)
                pa = (0, 0)
                pwb = tuple(cnt[hull[iwb]][0][0])
                phb = (0, 0)
                for k in range(len(hull)):
                    phb = tuple(cnt[hull[iwb - k]][0][0])
                    if distance(pwb, phb) > 40:
                        break
                slope_wp = (phb[1] - pwb[1]) / (phb[0] - pwb[0])

                for k in range(len(hull)):
                    pb = tuple(cnt[hull[ifp + k]][0][0])
                    if distance(p0, pb) > 40:
                        break
                # measure finger width
                # draw a line parallel with the finger
                for k in range(len(hull)):
                    pa = tuple(cnt[hull[ifp - k]][0][0])
                    if distance(p0, pa) > 40:
                        break
                cv.line(blur, p0, pa, [127, 255, 255], 2)
                # draw a line perpendicular to the finger
                c0, c1, d0, d1 = getPerpCoord(p0[0], p0[1], pa[0], pa[1], 50)
                tmp = np.zeros_like(blur, np.uint8)
                cv.line(tmp, (c0, c1), (d0, d1), [255, 255, 255], 5)
                ref = np.zeros_like(blur, np.uint8)
                ref = cv.drawContours(ref, [cnt], -1, (255, 255, 255), 1)
                # find intercept points of the line and the contour
                (x_intercept, y_intercept) = np.nonzero(np.logical_and(tmp, ref))
                # calculate distances between the points and the line
                p1 = np.array(p0)
                p2 = np.array(pa)
                dists = list()
                for m in range(len(x_intercept)):
                    p3 = np.array([y_intercept[m], x_intercept[m]])
                    cv.circle(blur, tuple(p3), 2, (0, 255, 255), -1)
                    dist = abs(np.cross(p2 - p1, p1 - p3) / np.linalg.norm(p2 - p1))
                    if dist > 8 and dist < 40:
                        dists.append(dist)
                finger_width = np.nanmean(dists)

                slope_fp = (p0[1] - pb[1]) / (p0[0] - pb[0])
                # print(slope_wp, slope_fp, slope_wp-slope_fp)
                cv.circle(temp0, p0, 10, (127, 255, 255), -1)
                cv.circle(temp0, pb, 10, (127, 255, 255), -1)
                cv.circle(temp0, pwb, 10, (127, 255, 255), -1)
                cv.circle(temp0, phb, 10, (127, 255, 255), -1)
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                h_rotated = distance(box[-1], box[-2])

                cv.rectangle(temp0, (x, y), (x + w, y + h), 255, 2)
                cv.drawContours(temp0, [box], 0, (127, 0, 255), 2)
                # cv.rectangle(temp0, (int(x-w/2), y), (int(x+w), y+h), 255, 2)

                idy = 0
                indextip = (0,0)
                for cp in cnt:
                    if abs(cp[0][0] - p0[0]) < 10 and cp[0][1] > p0[1]:
                        if cp[0][1] > idy:
                            idy = cp[0][1]
                            indextip = tuple(cp[0])
                cv.circle(blur, p0, 10, (127, 255, 255), -1)

                th0 = th0.astype(np.bool)

                wr = np.sum(th0.transpose()[0]) / len(th0)
                ar_full = np.sum(np.sum(th0)) / (len(th0) * len(th0.transpose())) * 1000

                th0[indextip[1]:] = 0
                ar = np.sum(np.sum(th0)) / (len(th0) * len(th0.transpose())) * 1000

                ph_d = (0, 0)
                dmax = 0
                for k in range(len(hull)):
                    ptmp = np.array(cnt[hull[k]][0][0])
                    dist = abs(np.cross(np.array(pwb) - np.array(indextip), np.array(indextip) - ptmp) /
                               np.linalg.norm(np.array(pwb) - np.array(indextip)))
                    if dist > dmax:
                        dmax = dist
                        ph_d = tuple(ptmp)
                tp_model.append([i, j, tuple(indextip), tuple(ph), ph_d, h, h_rotated, dmax, finger_width])
                if i == 0:
                    fig, ([ax0, ax1, ax2]) = plt.subplots(1, 3)
                    ax0.imshow(blur, cmap='seismic')
                    ax1.imshow(th0, cmap='seismic')
                    ax2.imshow(temp0, cmap='seismic')
                    plt.title(str(i) + '_' + str(j))
                    plt.show()

        # df_model = pd.DataFrame(tp_model,columns=['i', 'j', 'fingertip', 'ph', 'ph_d', 'h', 'h_rotate', 'dmax', 'finger_width'])
        # print(df_model)
        # df_model.to_csv('correction_model.csv')



    finally:
        cv.destroyAllWindows()
