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
stop_event = threading.Event()
data_reader0 = SerialReader(stop_event, q0, 'COM17')
data_reader0.start()
# data_reader1 = SerialReader(stop_event, q1, 'COM18')
# data_reader1.start()
re_size = (24*10, 32*10)
path = './trackpad_model_data/'

if __name__ == '__main__':
    try:
        fig, (ax0) = plt.subplots(1, 1)
        im0 = ax0.imshow(np.random.uniform(low=20, high=35, size=re_size), cmap='seismic')
        plt.tight_layout()
        plt.ion()
        xrg = 7
        yrg = 6
        xcur = 0
        ycur = 0
        tcnt = 0
        time_start = time.monotonic()
        t = 4
        cimg = [[[] for _ in range(yrg)] for _ in range(xrg)]
        # time.sleep(5)
        while True:
            if xcur > xrg:
                break
            time0, temp0 = q0.get()
            img0 = np.array([[0] * 32 for _ in range(24)], np.uint8)
            for i, x in enumerate(temp0):
                row = i // 32
                col = 31 - i % 32
                img0[int(row)][int(col)] = x

            time_loop = round(time.monotonic() - time_start, 2)
            tcnt = int(time_loop) // t
            xcur = tcnt // yrg
            ycur = tcnt % yrg
            filename = str(xcur) + '_' + str(ycur) + '_' + str(time_loop)
            if int(time_loop) % t < 3:
                print('MOVE! x:{}, y:{}'.format(xcur, ycur))
            else:
                print('STAY! x:{}, y:{}'.format(xcur, ycur))
                cimg[xcur][ycur].append(temp0)
                # cv.imwrite(path+filename+'.jpg', img0)
            img0 = cv.flip(img0, 0)
            img0 = cv.resize(img0, re_size, interpolation=cv.INTER_CUBIC)
            blur = cv.GaussianBlur(img0, (25, 25), 0)
            im0.set_array(blur)
            plt.pause(0.001)
        plt.close()
        with open(path + 'temps_model_flat'
                         '.pkl', 'wb') as file:
            pickle.dump(cimg, file)

    finally:
        cv.destroyAllWindows()
        stop_event.set()
        data_reader0.join()
        data_reader0.clean()
        # data_reader1.join()
        # data_reader1.clean()
