import serial
import matplotlib

matplotlib.use('TkAgg')  # MUST BE CALLED BEFORE IMPORTING plt
from matplotlib import pyplot as plt
import queue
import threading
from matplotlib import animation
import seaborn as sns
import numpy as np
import time
# import MLX90640
import cv2 as cv
from collections import deque
import math
import colorsys
import os
import pickle
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import argparse


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
                            self.signal.put([time.time(), self.temp[:-1]])
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
            j = ((data[i + 1] << 8) & 0xFF00) + data[i]
            self.temp[int(i / 2)] = j / 100.0
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

class symbol_detector():
    def __init__(self, com, rtplot):
        self.fsize = (24 * 10, 32 * 10)
        self.fsize1 = (32 * 10, 24 * 10)
        # self.fsize = self.fsize1
        self.flag_rtplot = rtplot
        self.com = com
        self.center = (0, 0)
        self.reader = None
        self.tempq = queue.Queue()
        self.stop_event = threading.Event()
        self.img = np.array([[1] * 32 for _ in range(24)], np.uint8)
        self.buttons_value = [0, 0, 0, 0]
        self.slides_value = [0, 0]
        self.indextip = (0, 0)
        self.handback = (0, 0)
        self.wrist = (0, 0)
        self.im = list()
        self.HuM_symbol = list()
        self.symbol_ims = dict()
        self.init_HuM()

    def initserial(self):
        self.reader = SerialReader(self.stop_event, self.tempq, self.com)
        self.reader.start()

    def clean(self):
        self.stop_event.set()
        self.reader.clean()
        self.reader.clean()

    def run(self):
        if self.flag_rtplot:
            self.rtplot()
        while not self.stop_event.is_set():
            time0, temp_raw = self.tempq.get()
            temp_scale = self.colorscale(temp_raw, np.min(temp_raw), np.max(temp_raw))
            # assemble image
            for i, x in enumerate(temp_scale):
                row = i // 32
                col = 31 - i % 32
                self.img[int(row)][int(col)] = x
            # resize, blur, and binarize image
            img = self.img
            # img[img < 0.3*256] = 0
            img = cv.resize(img, self.fsize1, interpolation=cv.INTER_CUBIC)
            img = cv.flip(img, 0)
            blur = cv.GaussianBlur(img, (31, 31), 0)
            ret, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

            dmin = 10000
            result = 'None'
            d_HuMs = dict()
            for sym, im in self.symbol_ims.items():
                d_HuM = cv.matchShapes(th, im, cv.CONTOURS_MATCH_I3, 0)
                d_HuMs[sym] = d_HuM
                if d_HuM < dmin:
                    dmin = d_HuM
                    result = sym
            print('{} '.format(d_HuMs))
            print('recognized as {} with distance {}'.format(result, dmin))
            # ret, th = cv.threshold(blur, np.amax(blur) * 0.6, 255, cv.THRESH_BINARY)
            # use the largest contour as the hand contour
            # contours, hierarchy = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # cv2.RETR_TREE
            # areas = [cv.contourArea(c) for c in contours]
            # if len(areas) == 0:
            #     continue
            # max_index = np.argmax(areas)
            # cnt = contours[max_index]
            #

            if self.flag_rtplot:
                cv.circle(blur, self.indextip, 5, (127, 255, 255), -1)
                cv.circle(blur, self.handback, 5, (127, 255, 255), -1)
                cv.circle(blur, self.wrist, 5, (127, 255, 255), -1)
                self.im[0].set_array(blur)
                self.im[0].set_array(blur)
                self.im[2].set_array(th)
                plt.pause(0.001)

    def init_HuM(self):
        filenames = [filename for filename in os.listdir('./heatlabel/') if filename.endswith('.jpg')]
        for filename in filenames:
            im = cv.imread('./heatlabel/'+filename, cv.IMREAD_GRAYSCALE)  # 直接读取为灰度图
            im = cv.resize(im, self.fsize1, interpolation=cv.INTER_CUBIC)
            im = cv.GaussianBlur(im, (31, 31), 0)
            _, im = cv.threshold(im, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            self.symbol_ims[filename.split('.')[0]] = im
            self.HuM_symbol.append(self.HuM(im))

    def HuM(self, im):
        # Calculate Moments
        moment = cv.moments(im)
        # Calculate Hu Moments
        huMoments = cv.HuMoments(moment)
        re_huMoments = []
        for i in range(0, 7):
            re_huMoments.append(-np.sign(huMoments[i][0]) * np.log10(abs(huMoments[i][0])))
        return re_huMoments

    def rtplot(self):
        fig, ([ax0, ax1], [ax2, ax3], [ax4, ax5]) = plt.subplots(3, 2)
        im0 = ax0.imshow(np.random.uniform(low=0, high=255, size=self.fsize), cmap='seismic')
        im1 = ax1.imshow(np.random.uniform(low=0, high=255, size=self.fsize), cmap='seismic')
        im2 = ax2.imshow(np.random.uniform(low=0, high=1, size=self.fsize), cmap=plt.cm.gray)
        im3 = ax3.imshow(np.random.uniform(low=0, high=255, size=self.fsize), cmap='seismic')
        im4 = ax4.imshow(np.random.uniform(low=0, high=1, size=self.fsize), cmap=plt.cm.gray)
        im5 = ax5.imshow(np.random.uniform(low=0, high=1, size=self.fsize), cmap=plt.cm.gray)
        self.im = [im0, im1, im2, im3, im4, im5]
        plt.tight_layout()
        plt.ion()

    def colorscale(self, datas, minc, maxc):
        data_scales = list()
        for data in datas:
            data_scale = int(256 * (data - minc) / (maxc - minc))
            # if data_scale < 0.5 * 256:
            if data_scale < 0:
                data_scale = 0
            elif data_scale > 255:
                data_scale = 255
            data_scales.append(data_scale)
        # print(data, data_scale)
        return data_scales

    def getPerpCoord(self, aX, aY, bX, bY, length):
        vX = bX - aX
        vY = bY - aY
        # print(str(vX)+" "+str(vY))
        if (vX == 0 or vY == 0):
            return 0, 0, 0, 0
        mag = math.sqrt(vX * vX + vY * vY)
        vX = vX / mag
        vY = vY / mag
        temp = vX
        vX = 0 - vY
        vY = temp
        pX = aX + (bX - aX) * 0.1
        pY = aY + (bX - aX) * 0.1
        cX = pX + vX * length
        cY = pY + vY * length
        dX = pX - vX * length
        dY = pY - vY * length
        return int(cX), int(cY), int(dX), int(dY)

    def pl_distance(self, p1, p2, p3):
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        return abs(np.cross(p2 - p1, p1 - p3) / np.linalg.norm(p2 - p1))

    def distance(self, point1, point2=None):
        if point2 is not None:
            return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
        else:
            return math.sqrt(point1[0] ** 2 + point1[1] ** 2)


if __name__ == '__main__':
    sd = symbol_detector(com='COM16', rtplot=True)
    try:
        sd.initserial()
        sd.run()
    finally:
        sd.clean()