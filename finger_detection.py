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
        if data_scale < 0.5*256:
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
            dist = math.sqrt(epind[0]**2 + (framesize[1]-epind[1])**2)
        elif wedge == 6:
            dist = math.sqrt((framesize[0]-epind[0])**2 + (framesize[1]-epind[1])**2)
        elif wedge == 7:
            dist = math.sqrt((framesize[0]-epind[0])**2 + epind[1]**2)
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

q0 = queue.Queue()
q1 = queue.Queue()
stop_event = threading.Event()
data_reader0 = SerialReader(stop_event, q0, 'COM16')
data_reader0.start()
data_reader1 = SerialReader(stop_event, q1, 'COM18')
data_reader1.start()
re_size = (24*5, 32*5)

if __name__ == '__main__':
    try:
        fig, ([ax0, ax1], [ax2, ax3], [ax4, ax5]) = plt.subplots(3, 2)
        # im1 = ax1.imshow(np.random.uniform(low=22, high=32, size=(20, 36)), vmin=20, vmax=36, cmap='jet')
        #                  # interpolation='lanczos')
        # im0 = ax0.imshow(np.random.uniform(low=22, high=32, size=(20, 36)), vmin=20, vmax=36, cmap='jet')
        #                  # interpolation='lanczos')
        im0 = ax0.imshow(np.random.uniform(low=0, high=255, size=re_size), cmap='seismic')
        im1 = ax1.imshow(np.random.uniform(low=0, high=255, size=re_size), cmap='seismic')
        im2 = ax2.imshow(np.random.uniform(low=0, high=1, size=re_size), cmap=plt.cm.gray)
        im3 = ax3.imshow(np.random.uniform(low=0, high=1, size=re_size), cmap=plt.cm.gray)
        im4 = ax4.imshow(np.random.uniform(low=0, high=1, size=re_size), cmap=plt.cm.gray)
        im5 = ax5.imshow(np.random.uniform(low=0, high=1, size=re_size), cmap=plt.cm.gray)
        plt.tight_layout()
        plt.ion()
        fgbg = cv.createBackgroundSubtractorMOG2(history=5, detectShadows=False)
        mask = np.array([[1] * 32 for _ in range(24)], np.uint8)
        cnt = 0
        # mlen = 10
        # bg_queue = deque(maxlen=mlen)
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
        while True:
            # if cnt % 20 == 0:
            #     mask = np.array([[255] * 32 for _ in range(24)], np.uint8)
            # cnt += 1
            time0, temp0 = q0.get()
            # temp1 = temp0
            time1, temp1 = q1.get()
            temp0 = colorscale(temp0, np.min(temp0), np.max(temp0))
            temp1 = colorscale(temp1, np.min(temp1), np.max(temp1))
            img0 = np.array([[0] * 32 for _ in range(24)], np.uint8)
            img1 = np.array([[0] * 32 for _ in range(24)], np.uint8)
            for i, x in enumerate(temp0):
                row = i // 32
                col = 31 - i % 32
                img0[int(row)][int(col)] = x
                img1[int(row)][int(col)] = temp1[i]
            # img0 = image_filter(img0)
            # img1 = image_filter(img1)
            # img0 = constrain(map_def(img0, np.amax(img0), np.amin(img0), 0, 255), 0, 255)
            # print(img0)
            img0 = cv.resize(img0, re_size, interpolation=cv.INTER_CUBIC)
            img1 = cv.resize(img1, re_size, interpolation=cv.INTER_CUBIC)
            img0 = cv.flip(img0, 0)
            img1 = cv.flip(img1, 0)

            # blur, opening, and erode
            blur0 = cv.GaussianBlur(img0, (25, 25), 0)
            blur1 = cv.GaussianBlur(img1, (25, 25), 0)
            # im0.set_array(blur0)
            # im1.set_array(blur1)
            # # thresh = np.max(blur)*0.9
            # # ret, th = cv.threshold(blur, thresh, 255, cv.THRESH_BINARY)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
            # ret, blur0 = cv.threshold(blur0, 100, 255, cv.THRESH_BINARY)
            # ret, blur1 = cv.threshold(blur1, 100, 255, cv.THRESH_BINARY)

            ret, th0 = cv.threshold(blur0, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
            # th0_erode = cv.erode(th0, kernel)
            ret, th1 = cv.threshold(blur1, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
            # th1_erode = cv.erode(th1, kernel)

            contours, hierarchy = cv.findContours(th0, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # cv2.RETR_TREE
            # blur0 = cv.drawContours(blur0, contours, -1, (0, 255, 0), 3)

            areas = [cv.contourArea(c) for c in contours]
            if areas:
                max_index = np.argmax(areas)
                cnt = contours[max_index]
                # Create a mask from the largest contour
                mask = np.zeros(th0.shape)
                cv.fillPoly(mask, [cnt], 1)
                # Use mask to crop data from original image
                th0 = np.multiply(th0, mask)
            if len(areas) > 1:
                cnt0_prev = contours[areas.index(np.sort(areas)[-2])]
            contours, hierarchy = cv.findContours(th1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # cv2.RETR_TREE
            # blur0 = cv.drawContours(blur0, contours, -1, (0, 255, 0), 3)

            areas = [cv.contourArea(c) for c in contours]
            if areas:
                max_index = np.argmax(areas)
                cnt = contours[max_index]
                # Create a mask from the largest contour
                mask = np.zeros(th1.shape)
                cv.fillPoly(mask, [cnt], 1)
                # Use mask to crop data from original image
                th1 = np.multiply(th1, mask)
            if len(areas) > 1:
                cnt1_prev = contours[areas.index(np.sort(areas)[-2])]
            im2.set_array(th0)
            im3.set_array(th1)

            th0 = th0.astype(np.bool)
            th1 = th1.astype(np.bool)
            th0_s = morphology.skeletonize(th0)
            th1_s = morphology.skeletonize(th1)
            im4.set_array(th0_s)
            im5.set_array(th1_s)
            # th0m, distance0 = morphology.medial_axis(th0, return_distance=True)
            # th1m, distance1 = morphology.medial_axis(th1, return_distance=True)
            # th0_m = distance0 * th0m
            # th1_m = distance1 * th1m
            # im4.set_array(th0_m)
            # im5.set_array(th1_m)

            # im3.set_array(th1_s)
            # th0_s = morphology.thin(th0, max_iter=17)
            # th1_s = morphology.thin(th1, max_iter=17)

            # im2.set_array(th0_s)
            # bp0 = branchPoints(th0_s)
            # bp0_ind = np.where(bp0 == 1)
            ep0 = endPoints(th0_s)
            ep0_ind = np.where(ep0 == 1)
            # print(list(ep0_ind))
            if len(ep0_ind[0]) >= 1:
                wedge0 = wristedge(th0, wedge0_prev)
                wedge0_prev = wedge0
                indextip0 = find_fingertip(wedge0, ep0_ind, re_size, indextip0_prev)
                # print(cv.pointPolygonTest(cnt0_prev,(indextip0[1], indextip0[0]), False))
                if len(cnt0_prev) > 1 and cv.pointPolygonTest(cnt0_prev,(indextip0[0], indextip0[1]), False) >= 0:
                        indextip0 = indextip0_prev
            else:
                indextip0 = indextip0_prev

            # cv.putText(blur0, '*', (50, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            # im2.set_array(blur0)


            # bp1 = branchPoints(th1_s)
            # bp1_ind = np.where(bp1 == 1)
            ep1 = endPoints(th1_s)
            ep1_ind = np.where(ep1 == 1)
            if len(ep1_ind[0]) >= 1:
                wedge1 = wristedge(th1, wedge1_prev)
                wedge1_prev = wedge1
                indextip1 = find_fingertip(wedge1, ep1_ind, re_size, indextip1_prev)
                if len(cnt1_prev) > 1 and cv.pointPolygonTest(cnt1_prev, (indextip1[0], indextip1[1]), False) >= 0:
                    indextip1 = indextip1_prev
            else:
                indextip1 = indextip1_prev

            # center = [-1, -1]
            # contours, hierarchy = cv.findContours(th0_erode, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # cv2.RETR_TREE
            # areas = [cv.contourArea(c) for c in contours]
            # if areas:
            #     # print(areas)
            #     max_index = np.argmax(areas)
            #     cnt = contours[max_index]
            #     x, y, w, h = cv.boundingRect(cnt)
            #     center = [x + w / 2, y + h / 2]
            #
            # indextip0 = find_indextip((np.array([center[0]]), np.array([center[1]])), ep0_ind, indextip0_prev)
            # indextip0, bp_prev = find_indextip(bp0_ind, ep0_ind, indextip0_prev, bp_prev)
            # indextip1 = find_indextip(bp1_ind, ep1_ind, indextip1_prev)
            indextip0_prev = indextip0
            indextip1_prev = indextip1

            print('ring0 tip in {}, ring1 tip is {}, edge is {}'.format(indextip0, indextip1, [wedge0_prev, wedge1_prev]))
            # ep1 = endPoints(th0_s)
            # im2.set_array(
            # im3.set_array(branchPoints(th1_s))
            # im4.set_array(endPoints(th0_s))
            # im5.set_array(endPoints(th1_s))

            # th0_s = np.array(th0_s, np.uint8)
            # indextip0 = trackFingertip(th0_s, indextip0_prev)
            # indextip0_prev = indextip0
            # if indextip0 is not None:
            blur0 = cv.circle(blur0, (indextip0[1], indextip0[0]), 5, (0, 127, 255), -1)
            # cv.putText(blur0, '*', (indextip0[1], indextip0[0]), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            # im2.set_array(th0_s)
            # indextip1 = trackFingertip(th_erode1, indextip1_prev)
            # indextip1_prev = indextip1
            # if indextip1 is not None:

            blur1 = cv.circle(blur1, (indextip1[1], indextip1[0]), 5, (0, 127, 255), -1)
            # cv.putText(blur1,'*',(indextip1[1], indextip1[0]), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2,cv.LINE_AA)

            im0.set_array(blur0)
            im1.set_array(blur1)
            # im3.set_array(th_erode1)
            # plt.pause(0.001)

        # # plt.ioff()
        # plt.show()
        # fig, (ax1, ax0) = plt.subplots(1, 2)
        # im1 = ax1.imshow(np.random.uniform(low=22, high=32, size=(8, 8)), vmin=22, vmax=32, cmap='jet', interpolation='lanczos')
        # im0 = ax0.imshow(np.random.uniform(low=22, high=32, size=(8,8)), vmin = 22, vmax = 32, cmap='jet', interpolation='lanczos')
        # plt.tight_layout()
        # plt.ion()
        # while True:
        #     [frame1, frame0] = q.get()
        #     im1.set_array(np.reshape(frame1, (8, 8)))
        #     im0.set_array(np.reshape(frame0, (8, 8)))
        #     # plt.draw()
            plt.pause(0.001)
        # plt.ioff()
        # plt.show()
    # except KeyboardInterrupt:
    #     cv.imwrite('./fingertip.jpg', img0)
    # except ValueError:
    #     print(th0)
    finally:
        cv.destroyAllWindows()
        stop_event.set()
        data_reader0.clean()
        data_reader0.clean()
        data_reader1.join()
        data_reader1.join()
