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
from scipy.spatial.distance import cdist, cosine
from shape_context import ShapeContext
import msvcrt

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
        self.hu_size = (20 * 10, 20 * 10)
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
        self.mlen = 8
        self.q_temp = deque(maxlen=self.mlen)
        self.HuM_symbol = dict()
        self.symbol_ims = dict()
        self.sc = None
        self.base = list()
        self.sym_names = list()
        self.test = None
        self.n_points = 15
        self.cnt = None
        self.img_cnt = 0
        self.save_flag = -1
        self.area_ratio = list()
        self.HuM_save = dict()
        self.name = 'None'
        self.start_flag = 0
        self.start_time = time.time()
        self.time_duration = list()
        self.init_HuM()
        # self.init_sc()

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
            self.q_temp.append(temp_raw)
            if len(list(self.q_temp)) == self.mlen:
                temp0 = np.average(np.array(self.q_temp), weights=range(1, self.mlen + 1), axis=0)
            else:
                temp0 = temp_raw
            temp_scale = self.colorscale(temp0, np.min(temp0), np.max(temp0))
            # if np.max(temp0) < 30:
            #     continue
            # assemble image
            for i, x in enumerate(temp_scale):
                row = i // 32
                col = 31 - i % 32
                self.img[int(row)][int(col)] = x
            # resize, blur, and binarize image
            img = self.img.copy()
            # img[img < 0.2*256] = 0
            img = cv.resize(img, self.fsize1, interpolation=cv.INTER_CUBIC)
            img = cv.flip(img, 0)
            blur = cv.GaussianBlur(img, (31, 31), 0)
            ret, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            # kernel = np.ones((5, 5), np.uint8)
            # th = cv.erode(th, kernel)
            # self.test = self.parse_img(th)
            # print(self.match(np.array(self.base), self.test))

            # pointst = self.sc.get_points_from_img(th, 15)
            # self.test = self.sc.compute(pointst).flatten()
            # res_min = 10000
            # result = '0'
            # for i, base in enumerate(self.base):
            #     # print(base.shape, descriptor.shape)
            #     # res = cdist(np.array([base]), np.array([descriptort]), metric="cosine")
            #     res = cdist(base, self.test)[0]
            #     if res < res_min:
            #         res_min = res
            #         result = self.sym_names[i]
            # print(result)
            # self.test = self.parse_img(th)
            # self.match(self.base, self.test)

            dmin = 10000
            result = 'None'
            d_HuMs = dict()
            contours, hierarchy = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            areas = [cv.contourArea(c) for c in contours]
            if len(areas) == 0:
                continue
            max_index = np.argmax(areas)
            self.cnt = contours[max_index]
            (x, y, w, h) = cv.boundingRect(self.cnt)
            # print(areas[max_index] / (self.fsize1[0] * self.fsize1[1]))
            area_ratio = w*h / (self.fsize1[0] * self.fsize1[1])
            if area_ratio > 0.6 and self.start_flag == 0:
                self.start_flag = 1
                self.start_time = time.time()
            if self.start_flag == 1:
                self.area_ratio.append(area_ratio)
            if 0.1 <area_ratio < 0.6 and len(self.area_ratio) > 8 and np.std(self.area_ratio[-8:]) < 0.015 and np.max(temp0) > 30:
                print('{} image {} saved!'.format(self.name, self.img_cnt))
                # cv.imwrite('./heatlabel/' + self.name + '_' + str(self.img_cnt) + '_xxh_s2_2.jpg', self.img)
                self.img_cnt += 1
                self.area_ratio = list()
                self.start_flag = 0
                self.time_duration.append(time.time()-self.start_time)
            # print(area_ratio)
            # if 0.2 < area_ratio < 0.5 and self.save_flag == 1:
            #     print('{} image saved!'.format(str(self.name)))
            #     cv.imwrite('./heatlabel/' + str(self.name) + '_' + str(self.img_cnt) + '.jpg', self.img)
            #     self.img_cnt += 1
            # if msvcrt.kbhit():
            #     key = msvcrt.getch().decode()
            #     if key is 's':
            #         self.save_flag = -self.save_flag
            #         if self.save_flag == 1:
            #             'start saving!'
            #     if key is 'n':
            #         self.name = self.name + 1
            #         #     break
            #         # elif key is 'm':
            #         #     self.img_cnt += 1
            #         #     break

            th_crop = cv.resize(th[y:y + h, x:x + w], (w*5, h*5), interpolation=cv.INTER_CUBIC)
            # # find contour again
            # contours, hierarchy = cv.findContours(th_crop, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # areas = [cv.contourArea(c) for c in contours]
            # if len(areas) == 0:
            #     continue
            # max_index = np.argmax(areas)
            # approx = cv.approxPolyDP(contours[max_index], 0.01 * cv.arcLength(contours[max_index], True), True)
            approx = contours[max_index]
            if msvcrt.kbhit():
                key = msvcrt.getch().decode()
                if key is '1':
                    self.name = 'play'
                    self.img_cnt = 0
                elif key is '2':
                    self.name = 'stop'
                    self.img_cnt = 0
                elif key is '3':
                    self.name = 'qus'
                    self.img_cnt = 0
                elif key is '4':
                    self.name = 'lock'
                    self.img_cnt = 0
                elif key is '5':
                    self.name = 'search'
                    self.img_cnt = 0
                elif key is '6':
                    self.name = 'light'
                    self.img_cnt = 0
                elif key is '7':
                    self.name = 'plus'
                    self.img_cnt = 0
                elif key is '8':
                    self.name = 'minus'
                    self.img_cnt = 0
                print('start {} collection'.format(self.name))
                # if key is 'm':
                # print('{} image saved!'.format(name))
                # cv.imwrite('./heatlabel/' + name + '_' + str(self.img_cnt) + '.jpg', self.img)
                # self.img_cnt += 1
            # for sym, im in self.symbol_ims.items():
            #     # th_crop = cv.resize(th[y:y + h, x:x + w], im.shape, interpolation=cv.INTER_CUBIC)
            #     d_HuM = cv.matchShapes(th_crop, im, cv.CONTOURS_MATCH_I2, 0)
            #     d_HuMs[sym] = d_HuM
            #     self.dist[sym] += d_HuM
            #     if d_HuM < dmin:
            #         dmin = d_HuM
            #         result = sym
            # print('{} '.format(d_HuMs))
            # print('recognized as {} with distance {}'.format(result, dmin))

            if self.flag_rtplot:
                # cv.circle(blur, self.indextip, 5, (127, 255, 255), -1)
                # cv.circle(blur, self.handback, 5, (127, 255, 255), -1)
                # cv.circle(blur, self.wrist, 5, (127, 255, 255), -1)
                cv.drawContours(blur, [self.cnt], 0, color=(255, 255, 255))
                cv.drawContours(th_crop, [approx], 0, color=(255, 255, 255))
                (x, y, w, h) = cv.boundingRect(self.cnt)
                cv.rectangle(blur, (x, y), (x+w, y+h), 255, 2)

                self.im[0].set_array(blur)
                self.im[1].set_array(blur)
                self.im[2].set_array(th)
                self.im[4].set_array(th_crop)
                self.im[4].set_array(th_crop)
                plt.pause(0.001)

    def init_HuM(self):
        filenames = [filename for filename in os.listdir('./heatlabel/') if filename.endswith('.jpg') and filename.startswith('img')]

        for filename in filenames:
            im = cv.imread('./heatlabel/'+filename, cv.IMREAD_GRAYSCALE)  # 直接读取为灰度图
            im = cv.resize(im, self.fsize1, interpolation=cv.INTER_CUBIC)
            im = cv.GaussianBlur(im, (31, 31), 0)
            _, im = cv.threshold(im, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            contours, hierarchy = cv.findContours(im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            areas = [cv.contourArea(c) for c in contours]
            if len(areas) == 0:
                continue
            max_index = np.argmax(areas)
            self.cnt = contours[max_index]
            (x, y, w, h) = cv.boundingRect(self.cnt)
            th_crop = cv.resize(im[y:y + h, x:x + w],  self.hu_size, interpolation=cv.INTER_CUBIC)
            # contours, hierarchy = cv.findContours(th_crop, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # areas = [cv.contourArea(c) for c in contours]
            # if len(areas) == 0:
            #     continue
            # max_index = np.argmax(areas)
            # approx = contours[max_index]
            # approx = cv.approxPolyDP(contours[max_index], 0.01 * cv.arcLength(contours[max_index], True), True)
            self.symbol_ims[filename.split('.')[0]] = th_crop
            self.HuM_symbol[filename.split('.')[0]] = self.HuM(th_crop)
            # self.HuM_symbol.append(self.HuM(im))

    def HuM(self, im):
        # contours, hierarchy = cv.findContours(im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # areas = [cv.contourArea(c) for c in contours]
        # max_index = np.argmax(areas)
        # cnt = contours[max_index]
        # self.cnt = cnt
        # (x, y, w, h) = cv.boundingRect(cnt)
        # im = cv.resize(im[x:x+w][y:y+h], self.fsize1, interpolation=cv.INTER_CUBIC)
        # Calculate Moments
        moment = cv.moments(im)
        # Calculate Hu Moments
        huMoments = cv.HuMoments(moment)
        re_huMoments = []
        for i in range(0, 6):
            re_huMoments.append(-np.sign(huMoments[i][0]) * np.log10(abs(huMoments[i][0])))
        return re_huMoments

    def init_sc(self):
        self.sc = ShapeContext()
        filenames = [filename for filename in os.listdir('./heatlabel/') if filename.endswith('.jpg') and not filename.startswith('img')]
        kernel = np.ones((5, 5), np.uint8)

        for n, filename in enumerate(filenames):
            # plt.figure()
            im = cv.imread('./heatlabel/'+filename, cv.IMREAD_GRAYSCALE)  # 直接读取为灰度图
            # im = cv.bitwise_not(im)
            im = cv.resize(im, self.fsize1, interpolation=cv.INTER_CUBIC)
            blur = cv.GaussianBlur(im, (31, 31), 0)
            ret, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
            # th = cv.dilate(th, kernel, iterations=1)
            # plt.imshow(th)
            self.base.append(self.parse_img(th))
            # points = self.sc.get_points_from_img(th, 15)
            # descriptor = self.sc.compute(points).flatten()
            # self.base.append(descriptor)
            self.sym_names.append(filename.split('.')[0])
        # plt.show()
        # imt = cv.imread('./heatlabel/search.jpg', cv.IMREAD_GRAYSCALE)  # 直接读取为灰度图
        # # imt = cv.bitwise_not(imt)
        # imt = cv.resize(imt, self.fsize1, interpolation=cv.INTER_CUBIC)
        # blurt = cv.GaussianBlur(imt, (31, 31), 0)
        # ret, tht = cv.threshold(blurt, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        # self.test = self.parse_img(tht)
        # # tht = cv.dilate(tht, kernel, iterations=1)
        # # ret, tht = cv.threshold(blurt, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # pointst = self.sc.get_points_from_img(tht, 15)
        # descriptort = self.sc.compute(pointst).flatten()
        # for i, base in enumerate(self.base):
        #     # print(base.shape, descriptor.shape)
        #     # res = cdist(np.array([base]), np.array([descriptort]), metric="cosine")
        #     res = cdist(np.array([base]), np.array([descriptort]))
        #     print(self.sym_names[i], res)
        # print(np.array(self.base).shape, self.test.shape)
        # print(self.match(np.array(self.base), self.test))

    def get_contour_bounding_rectangles(self, gray):
        """
          Getting all 2nd level bouding boxes based on contour detection algorithm.
        """
        contours, hierarchy = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        res = []
        for cnt in contours:
            (x, y, w, h) = cv.boundingRect(cnt)
            res.append((x, y, x + w, y + h))
        return res

    def get_contour_bounding_rectangles_largest(self, gray):
        """
          Getting all 2nd level bouding boxes based on contour detection algorithm.
        """
        contours, hierarchy = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        areas = [cv.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        self.cnt = cnt
        (x, y, w, h) = cv.boundingRect(cnt)
        # res = []
        # for cnt in contours:
        #     (x, y, w, h) = cv.boundingRect(cnt)
        #     res.append((x, y, x + w, y + h))
        return [(x, y, x + w, y + h)]

    def parse_img(self, img):
        # invert image colors
        # img = cv.bitwise_not(img)
        # _, img = cv.threshold(img, 128, 255, cv.THRESH_BINARY)
        # making numbers fat for better contour detectiion
        # kernel = np.ones((2, 2), np.uint8)
        # img = cv.dilate(img, kernel, iterations=1)

        # getting our numbers one by one
        rois = self.get_contour_bounding_rectangles_largest(img)
        # print(len(rois))
        grayd = cv.cvtColor(img.copy(), cv.COLOR_GRAY2BGR)
        nums = []
        for r in rois:
            grayd = cv.rectangle(grayd, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 1)
            nums.append((r[0], r[1], r[2], r[3]))
        # we are getting contours in different order so we need to sort them by x1
        nums = sorted(nums, key=lambda x: x[0])
        descs = []
        for i, r in enumerate(nums):
            if img[r[1]:r[3], r[0]:r[2]].mean() < 50:
                continue
            im = cv.resize(img[r[1]:r[3], r[0]:r[2]], self.fsize1, interpolation=cv.INTER_CUBIC)
            # blur = cv.GaussianBlur(im, (31, 31), 0)
            # ret, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            # points = self.sc.get_points_from_img(img[r[1]:r[3], r[0]:r[2]], 20)
            points = self.sc.get_points_from_img(im, self.n_points)
            descriptor = self.sc.compute(points).flatten()
            descs.append(descriptor)
        # print(descs)
        return np.array(descs[0])


    def match(self, base, current):
        """
          Here we are using cosine diff instead of "by paper" diff, cause it's faster
        """
        res = cdist(base, current.reshape((1, current.shape[0])), metric="cosine")
        # res = cdist(base, current, metric="cosine")

        idxmin = np.argmin(res.reshape(len(base)))
        result = self.sym_names[idxmin]
        return result, res.reshape(len(base))[idxmin]

    def rtplot(self):
        fig, ([ax0, ax1], [ax2, ax3], [ax4, ax5]) = plt.subplots(3, 2)
        im0 = ax0.imshow(np.random.uniform(low=0, high=255, size=self.fsize), cmap='seismic')
        im1 = ax1.imshow(np.random.uniform(low=0, high=255, size=self.fsize), cmap='seismic')
        im2 = ax2.imshow(np.random.uniform(low=0, high=1, size=self.fsize), cmap=plt.cm.gray)
        im3 = ax3.imshow(np.random.uniform(low=0, high=255, size=self.fsize), cmap='seismic')
        im4 = ax4.imshow(np.random.uniform(low=0, high=1, size=self.hu_size), cmap=plt.cm.gray)
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
    # except KeyboardInterrupt as e:

    finally:
        sd.clean()