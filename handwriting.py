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
from mpl_toolkits.mplot3d import Axes3D
import pickle
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

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
        data_scale = int(256 * (data - minc) / (maxc - minc))
        if data_scale < 0.5 * 256:
        # if data_scale < 0:
            data_scale = 0
        elif data_scale > 255:
            data_scale = 255
        data_scales.append(data_scale)
    # print(data, data_scale)
    return data_scales

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

def confine(val, min_val, max_val):
    return min(max_val, max(val, min_val))

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

def find_fingertip(wedge, ep_ind, framesize, indextip_prev):
    dmax = 0
    indextip = indextip_prev
    if len(ep_ind[0]) < 1:
        return indextip
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
    return (indextip[1], indextip[0])

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

def getfingertip_q(q, p, weights):
    q.append(p)
    lq = list(q)
    x=0
    y=0
    for i, item in enumerate(lq):
        x += item[0] * weights[i] / np.sum(weights)
        y += item[1] * weights[i] / np.sum(weights)
    return (int(x / len(lq)), int(y / len(lq)))

def log_transform(a):
    sign = a / abs(a)
    return np.log2(abs(a)) * sign

def distance(point1, point2=None):
    if point2 is not None:
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    else:
        return math.sqrt(point1[0] ** 2 + point1[1] ** 2)

# q0 = queue.Queue()
# q1 = queue.Queue()
# stop_event = threading.Event()
# data_reader0 = SerialReader(stop_event, q0, 'COM17')
# data_reader0.start()
# data_reader1 = SerialReader(stop_event, q1, 'COM18')
# data_reader1.start()
re_size = (24*10, 32*10)
plot_size = (32*10, 24*10)
path = './handwriting/'

if __name__ == '__main__':
    try:
        fig, ([ax0, ax1], [ax2, ax3], [ax4, ax5]) = plt.subplots(3, 2)
        # im1 = ax1.imshow(np.random.uniform(low=22, high=32, size=(20, 36)), vmin=20, vmax=36, cmap='jet')
        #                  # interpolation='lanczos')
        # im0 = ax0.imshow(np.random.uniform(low=22, high=32, size=(20, 36)), vmin=20, vmax=36, cmap='jet')
        #                  # interpolation='lanczos')
        im0 = ax0.imshow(np.random.uniform(low=0, high=255, size=re_size), cmap='seismic')
        im1 = ax1.imshow(np.random.uniform(low=0, high=255, size=re_size), cmap='seismic')
        im2 = ax2.imshow(np.random.uniform(low=0, high=1, size=plot_size), cmap=plt.cm.gray)
        im3 = ax3.imshow(np.random.uniform(low=0, high=255, size=plot_size), cmap='seismic')
        im4 = ax4.imshow(np.random.uniform(low=0, high=1, size=plot_size), cmap=plt.cm.gray)
        im5 = ax5.imshow(np.random.uniform(low=0, high=1, size=plot_size), cmap=plt.cm.gray)
        plt.tight_layout()
        plt.ion()

        mask = np.array([[1] * 32 for _ in range(24)], np.uint8)
        cnt = 0
        mlen = 1
        q_ft = deque(maxlen=mlen)
        q_ar = deque(maxlen=mlen)
        q_temp = deque(maxlen=mlen)
        indextip0 = [0, 0]
        indextip1 = [0, 0]
        indextip0_prev = [0, 0]
        indextip1_prev = [0, 0]
        wedge0_prev = -1
        wedge1_prev = -1
        bp_prev = [[0], [0]]
        cnt0_prev = [0]
        cnt1_prev = [0]
        # timg = cv.imread('fingertip.jpg')
        # print(np.shape(timg))
        cal_cnt = 30
        temp0_bg = [0] * 768
        temp1_bg = [0] * 768
        timestamp_prev = 0
        dtime = 0
        fullar_prev = 0
        par_prev = 0
        lift_flag = 0
        xcur = 50
        ycur = 50
        xcur_prev = 0
        ycur_prev = 0
        xkalman_prev = 0
        ykalman_prev = 0
        hp = 60
        yp = 50
        x_origin = 50
        ph_origin = (0, 0)
        x_re = 1
        dmax = 0
        h_correction = 0
        y_fts = list()
        h_fts = list()
        vflag = 0
        hflag = 0
        # with open(path + 'cal_matrix.pkl', 'rb') as file:
        #     [ret, mtx, dist, rvecs, tvecs] = pickle.load(file)
        # ctemps = []
        n_avg = 6
        name = 'zsy'
        fname = 'zsy_ques_3'
        with open(path + name + '/raw/'+fname+'_raw.pkl', 'rb') as file:
            ctemps_xv = pickle.load(file)
        gesture_cnt = 0
        gesture_raw = list()
        gesture_kalman = list()
        for gesture in ctemps_xv:
            # remove bad gesture
            if len(gesture) == 0:
                continue
            if gesture[-1][0] - gesture[0][0] < 0.3:
                print('too short!')
                continue
            # reinit kalman filter
            f = KalmanFilter(dim_x=4, dim_z=2)
            f.x = np.array([[5], [5], [0], [0]])

            f.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
            # f.G = np.array([[0, 0, 1, 0],
            #                 [0, 0, 0, 1]]).transpose()
            f.P *= 1000
            f.R *= 5
            f.Q = Q_discrete_white_noise(dim=4, dt=0.125, var=0.5)
            f.F = np.array([[1, 0, 0.125, 0],
                            [0, 1, 0, 0.125],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
            direction = list()
            direction_kalman = list()
            indextip0 = [0, 0]
            xcur = 50
            ycur = 50
            xcur_prev = 0
            ycur_prev = 0
            xkalman_prev = 0
            ykalman_prev = 0
            for enum, item in enumerate(gesture):
                time0 = item[0]
                temp0 = item[1]
                dtime = time0 - timestamp_prev
                timestamp_prev = time0
                temp0 = colorscale(temp0, np.min(temp0), np.max(temp0))
                img0 = np.array([[0] * 32 for _ in range(24)], np.uint8)
                for i, x in enumerate(temp0):
                    row = i // 32
                    col = 31 - i % 32
                    img0[int(row)][int(col)] = x
                img0[img0 < 140] = 0
                img0 = cv.resize(img0, plot_size, interpolation=cv.INTER_CUBIC)

                img0 = cv.flip(img0, 0)

                blur0 = cv.GaussianBlur(img0, (31, 31), 0)
                ret, th0 = cv.threshold(blur0, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

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
                mask = np.zeros_like(blur0)
                # cv.fillPoly(mask_ft, [cnt], 1)
                defects = cv.convexityDefects(cnt, hull)

                distmax = 0
                max_df = (0,0)
                temp1 = np.zeros_like(blur0)
                for l in range(defects.shape[0]):
                    s, e, fa, d = defects[l, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[fa][0])
                    cv.circle(temp1, far, 2, (127, 255, 255), -1)
                    if d > distmax:
                        distmax = d
                        max_df = far
                cv.circle(temp1, max_df, 5, (127, 255, 255), -1)
                im3.set_array(temp1)
                xmax = 0
                ymin = 10000
                xmin = 10000
                p0 = (0, 0)
                ifp = 0
                ph = 0
                iwb = 0
                temp0 = np.zeros_like(blur0)
                for i in range(len(hull)):
                    cv.circle(temp0, tuple(cnt[hull[i]][0][0]), 10, (127, 255, 255), -1)
                    if cnt[hull[i]][0][0][0] > xmax:
                        xmax = cnt[hull[i]][0][0][0]
                        p0 = tuple(cnt[hull[i]][0][0])
                        ifp = i
                    if cnt[hull[i]][0][0][1] < ymin:
                        ymin = cnt[hull[i]][0][0][1]
                        ph = tuple(cnt[hull[i]][0][0])
                    if cnt[hull[i]][0][0][0] <= xmin:
                        if cnt[hull[i]][0][0][0] == xmin:
                            if cnt[hull[i]][0][0][1] > cnt[hull[iwb]][0][0][1]:
                                iwb = i
                        xmin = cnt[hull[i]][0][0][0]
                        iwb = i-1
                    if i < len(hull) - 1:
                        cv.line(temp0, tuple(cnt[hull[i]][0][0]), tuple(cnt[hull[i + 1]][0][0]), [127, 255, 255], 3)
                    else:
                        cv.line(temp0, tuple(cnt[hull[i]][0][0]), tuple(cnt[hull[0]][0][0]), [127, 255, 255], 3)
                pb = (0,0)
                pwb = tuple(cnt[hull[iwb]][0][0])
                phb = (0,0)
                for i in range(len(hull)):
                    phb = tuple(cnt[hull[iwb - i]][0][0])
                    if distance(pwb, phb) > 40:
                        break
                slope_wp = (phb[1] - pwb[1]) / (phb[0] - pwb[0])

                for i in range(len(hull)):
                    pb = tuple(cnt[hull[ifp+i]][0][0])
                    if distance(p0, pb) > 40:
                        break
                pa = (0, 0)
                # measure finger width
                # draw a line parallel with the finger
                for i in range(len(hull)):
                    pa = tuple(cnt[hull[ifp - i]][0][0])
                    if distance(p0, pa) > 40:
                        break
                slope_ft_abs = (p0[1] - ph[1]) / (p0[0] - ph[0])
                slope_ft_center = (p0[1] - center[1]) / (p0[0] - center[0])
                slope_ft_phb = (p0[1] - phb[1]) / (p0[0] - phb[0])

                cv.line(blur0, p0, pa, [127, 255, 255], 5)
                # draw a line perpendicular to the finger
                c0, c1, d0, d1 = getPerpCoord(p0[0], p0[1], pa[0], pa[1], 50)
                tmp = np.zeros_like(blur0, np.uint8)
                cv.line(tmp, (c0, c1), (d0, d1), [255, 255, 255], 5)
                ref = np.zeros_like(blur0, np.uint8)
                ref = cv.drawContours(ref, [cnt], -1, (255, 255, 255), 1)
                # find intercept points of the line and the contour
                (x_intercept, y_intercept) = np.nonzero(np.logical_and(tmp, ref))
                # calculate distances between the points and the line
                p1 = np.array(p0)
                p2 = np.array(pa)
                dists = list()
                for m in range(len(x_intercept)):
                    p3 = np.array([y_intercept[m], x_intercept[m]])
                    cv.circle(blur0, tuple(p3), 2, (0, 255, 255), -1)
                    dist = abs(np.cross(p2 - p1, p1 - p3) / np.linalg.norm(p2 - p1))
                    if dist > 8 and dist < 40:
                        dists.append(dist)
                finger_width = np.nanmean(dists)

                # find the point of palm fist bottom
                for point in cnt:
                    if point[0][0] - ph[0] < 3:
                        phb = tuple(point[0])
                slope_fp = (p0[1] - phb[1]) / (p0[0] - phb[0])
                # print(slope_wp, slope_fp, slope_wp-slope_fp)
                cv.circle(temp0, p0, 10, (127, 255, 255), -1)
                cv.circle(temp0, pb, 10, (127, 255, 255), -1)
                cv.circle(temp0, pwb, 10, (0, 255, 255), -1)
                cv.circle(temp0, phb, 10, (0, 255, 255), -1)
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                box = np.int0(box)

                cv.rectangle(temp0, (x, y), (x+w, y+h), 255, 2)
                cv.drawContours(temp0, [box], 0, (127, 0, 255), 2)
                # cv.rectangle(temp0, (int(x-w/2), y), (int(x+w), y+h), 255, 2)

                idy = 0
                for cp in cnt:
                    if abs(cp[0][0] - p0[0]) < 10 and cp[0][1] > p0[1]:
                        if cp[0][1] > idy:
                            idy = cp[0][1]
                            indextip0 = tuple(cp[0])

                th0 = th0.astype(np.bool)
                # th0_s = morphology.skeletonize(th0)
                # im4.set_array(th0_s)

                wr = np.sum(th0.transpose()[0]) / len(th0)
                ar_full = np.sum(np.sum(th0)) / (len(th0) * len(th0.transpose())) * 1000

                th0[indextip0[1]:] = 0
                # for i, item in enumerate(th0):
                #     th0[i][:center[1]] = 0
                # th0 = np.multiply(th0, mask)
                im2.set_array(th0)
                ar = np.sum(np.sum(th0)) / (len(th0) * len(th0.transpose())) * 1000

                dr = 0
                fw = 0

                if len(dists) != 0:
                    fw = np.nanmin(dists)
                # print(slope_fp)
                # print('finger width:{0:.3f}, maxDefect:{1:.3f}, area ratio:{2:.3f}, full ar:{3:.3f}, wrist ratio:{4:.3f}'.format(fw, distmax,
                #                                                                                                 ar, ar_full, wr))
                indextip0_correct = 0

                dmax = 1
                for i in range(len(hull)):
                    ptmp = np.array(cnt[hull[i]][0][0])
                    dist = abs(np.cross(np.array(pwb) - np.array(indextip0), np.array(indextip0) - ptmp) /
                               np.linalg.norm(np.array(pwb) - np.array(indextip0)))
                    if dist > dmax:
                        dmax = dist
                        ph = tuple(ptmp)
                cv.circle(temp0, ph, 12, (127, 255, 255), -1)
                h_correction = h*abs(ph[0] - re_size[0]/2)/(re_size[0])*0.4
                # dmax = h + h_correction
                x_re = (60*216.7) / dmax

                # print(np.std(y_fts))
                # dmax = finger_width
                # if 10 < ph[0] and indextip0[0] < re_size[0] - 3:
                # if True:
                if enum == 1:
                    print('PUT DOWN! {}'.format(gesture_cnt))
                    gesture_cnt += 1
                    hp = h
                    lift_flag = 0
                    yp = indextip0[0]
                    x_origin = x_re
                    indextip0_correct = 0
                    ph_origin = ph
                    xkalman_prev = 0
                    ykalman_prev = 0
                elif enum > 1:
                    dr = (hp - dmax) / dmax
                    xcur = (x_re - x_origin)
                    # indextip0_correct = 3*(indextip0[0] - re_size[0] / 2) * (x_re - x_origin) / (x_origin)
                    ycur = 0.5*0.46*(indextip0[0] - yp)
                    x_mov = xcur - xcur_prev
                    y_mov = ycur - ycur_prev
                    direction.append([np.arctan2(y_mov, x_mov)*180/np.pi, math.sqrt(x_mov**2 + y_mov**2)])
                    xcur_prev = xcur
                    ycur_prev = ycur
                    # ycur = 20*(indextip0[0] + indextip0_correct - yp)/re_size[0] + ycur_prev
                    f.predict()
                    f.update([[xcur], [ycur]])
                    x_kalman_mov = f.x[0][0] - xkalman_prev
                    y_kalman_mov = f.x[1][0] - ykalman_prev
                    direction_kalman.append([np.arctan2(y_kalman_mov, x_kalman_mov)*180/np.pi,
                                             math.sqrt(x_kalman_mov**2 + y_kalman_mov**2)])
                    xkalman_prev = f.x[0][0]
                    ykalman_prev = f.x[1][0]
                    xcur = confine(f.x[0][0], -50, 50)
                    ycur = confine(f.x[1][0], -50, 50)
                    # x_tp = xcur
                    # y_tp = ycur
                    # xcur = confine(xcur, 0, 10)
                    # ycur = confine(ycur, 0, 10)

                im1.set_array(temp0)

                fullar_prev = ar_full
                par_prev = ar
                indextip0_prev = indextip0
                # d_est = (60*216.7)/(indextip0[1]-ph[1])
                # d_est = dr*indextip0[1] + indextip0[1]
                # print('Lift FLAG:{0}, par:{1:.3f}, far:{2:.3f}, vx:{3:.3f}, vy:{4:.3f}, v-par:{5:.3f}, v-far:{6:.3f}, distance:{7:.3f}'.
                #       format(lift_flag, ar, ar_full, v_x, v_lift, v_par, v_far, dr))

                blur0 = cv.circle(blur0, tuple(indextip0), 10, (127, 127, 255), -1)
                im0.set_array(blur0)
                # 3D plot

                # z_tp = map_def(indextip0[1], 0, re_size[1], 0, 10)
                # print(x_tp, y_tp, z_tp)
                # hl.set_data(xcur, ycur)
                # hl.set_3d_properties(z_tp)
                # print('Lift FLAG:{0}, x:{1:.3f},  h:{2:.3f}, y:{3:.3f}, y_correct:{4:.3f},, ycorrrect:{5:.3f}, hcorrect:{6:.3f}'
                #       .format(lift_flag, dmax,  h, ycur, indextip0[0]+indextip0_correct, indextip0_correct, h_correction))
                # print('Lift FLAG:{0}, slope_wp:{1:.3f},  slope_fp:{2:.3f}, delta:{3:.3f}, slope_ft_abs:{4:.3f}, slope_ft_center:{5:.3f}, slope_phb:{6:.3f}'
                #     .format(lift_flag, slope_wp, slope_fp, slope_wp-slope_fp, slope_ft_abs, slope_ft_center, slope_ft_phb))
                # plt.pause(0.08)
            # hist, avg = np.histogram(direction, 18)
            #
            # hist_kalman, avg_kalman = np.histogram(direction_kalman, 18)
            gesture_raw.append(direction)
            gesture_kalman.append(direction_kalman)
            # print(hist / np.sum(hist), avg)
            # print(hist_kalman / np.sum(hist_kalman), avg_kalman)
        with open(path + name + '/processed/'+fname+'_dir.pkl', 'wb') as file:
            pickle.dump([gesture_raw, gesture_kalman], file)
        # except KeyboardInterrupt:
        #     with open(path + 'x_movement.pkl', 'wb') as file:
        #         pickle.dump(cimg, file)
        # except ValueError:
        #     print(th0)
    finally:
        cv.destroyAllWindows()
