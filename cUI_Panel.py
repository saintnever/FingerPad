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

class UI_panel():
    def __init__(self, com, rtplot):
        self.fsize = (24*10, 32*10)
        self.fsize1 = (32*10, 24*10)
        # self.fsize = self.fsize1
        self.flag_rtplot = rtplot
        self.com = com
        self.center = (0,0)
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
        self.finger_slope = 0
        self.wrist_slope = 0
        self.slope = self.finger_slope - self.wrist_slope
        self.slope_prev = self.slope
        self.click_pos = 0
        self.flag_lift = 0
        self.lift_time = 0
        self.down_time = 0
        self.h_cal = -1
        self.h = 0
        self.slide = threading.Event()
        self.click = threading.Event()
        self.t_wait = 1.5
        self.slider_n = 0
        self.button_n = 0
        self.d = 1

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
            # img = img.transpose()
            img = cv.resize(img, self.fsize1, interpolation=cv.INTER_CUBIC)
            img = cv.flip(img, 0)
            # img = img.transpose()
            blur = cv.GaussianBlur(img, (31, 31), 0)
            # ret, th = cv.threshold(blur, np.amax(blur) * 0.8, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            ret, th = cv.threshold(blur, np.amax(blur) * 0.6, 255, cv.THRESH_BINARY)
            # use the largest contour as the hand contour
            contours, hierarchy = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # cv2.RETR_TREE
            areas = [cv.contourArea(c) for c in contours]
            if len(areas) == 0:
                continue
            max_index = np.argmax(areas)
            cnt = contours[max_index]
            # use convext hull to find the fingertip and hand back location
            hull = cv.convexHull(cnt, returnPoints=False)
            xmax = 0
            ymin = 10000
            xmin = 10000
            for i in range(len(hull)):
                # cv.circle(blur, tuple(cnt[hull[i]][0][0]), 6, (127, 255, 255), -1)
                # find is the hull point that has the largest x value (furthest from the wrist)
                if cnt[hull[i]][0][0][0] > xmax:
                    xmax = cnt[hull[i]][0][0][0]
                    self.indextip = tuple(cnt[hull[i]][0][0])
                # handback is the hull point that has the largest y value
                if cnt[hull[i]][0][0][1] < ymin:
                    ymin = cnt[hull[i]][0][0][1]
                    self.handback = tuple(cnt[hull[i]][0][0])
                if cnt[hull[i]][0][0][0] < xmin:
                    xmin = cnt[hull[i]][0][0][0]
                    self.wrist = tuple(cnt[hull[i]][0][0])
            # use the slope between handback and fingertip to determine whether the index finger is lifted
            self.finger_slope = (self.indextip[1] - self.handback[1]) / (self.indextip[0] - self.handback[0])
            # self.wrist_slope = (self.wrist[1] - self.handback[1]) / (self.wrist[0] - self.handback[0])
            # self.slope = self.finger_slope - self.wrist_slope
            self.click.clear()
            if self.finger_slope < 0.4:
                if self.flag_lift == 0:
                    if time.time() - self.lift_time < self.t_wait:
                        self.click.set()
                self.lift_time = time.time()
                self.flag_lift = 1
                self.slide.clear()
                self.slides_value = [0, 0]
                self.buttons_value = [0, 0, 0, 0]
            elif self.finger_slope > 0.5:
                # only update d when the finger first put down
                if self.flag_lift == 1:
                    self.flag_lift = 0
                    self.click_pos = self.indextip[0]
                    self.down_time = time.time()
                    # self.h = self.pl_distance(self.wrist, self.indextip, self.handback)
                    self.h = self.wrist[1] - self.handback[1]
                    if self.h_cal == -1:
                        # self.h_cal = self.pl_distance(self.wrist, self.indextip, self.handback)
                        self.h_cal = self.wrist[1] - self.handback[1]
                        self.finger_slope_cal = self.finger_slope
                    self.d = self.h / self.h_cal
                else:
                    if time.time() - self.down_time > self.t_wait and abs(self.indextip[0] - self.click_pos) > 10:
                        self.slide.set()
                        if self.d > 0.8:
                            self.slider_n = 0
                        else:
                            self.slider_n = 1
                    # cv.boundingRect(cnt)
            # determine the button clicked based on hand height ratio and fingertip location
            self.slope_prev = self.slope
            # if self.h_cal != 0:
            #     self.d = self.h/self.h_cal
            if self.click.is_set():
                print('Click!')
                if self.d > 0.8:
                    if self.click_pos < 190:
                        self.buttons_value[0] = 1
                        self.button_n = 0
                    else:
                        self.buttons_value[1] = 1
                        self.button_n = 1
                else:
                    if self.click_pos < 190 - 20:
                        self.buttons_value[3] = 1
                        self.button_n = 3
                    else:
                        self.buttons_value[2] = 1
                        self.button_n = 2
            if self.slide.is_set():
                print('Slide!')
                self.slides_value[self.slider_n] = self.indextip[0] - self.click_pos
                # self.slides_value[1] = self.indextip[0] - self.click_pos

            print('the fh slope is {0:.02f}, h is {1:.02f}, slider_value is {2:.02f}, lift_flag is {3}, button_No {4}, slider values {5}'
                  .format(self.finger_slope, self.h/self.h_cal, self.indextip[0], self.flag_lift, self.button_n, self.slides_value))

            if self.flag_rtplot:
                cv.circle(blur, self.indextip, 5, (127, 255, 255), -1)
                cv.circle(blur, self.handback, 5, (127, 255, 255), -1)
                cv.circle(blur, self.wrist, 5, (127, 255, 255), -1)
                self.im[0].set_array(blur)
                self.im[0].set_array(blur)
                self.im[2].set_array(th)
                plt.pause(0.001)

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

    def get_slider(self):
        return self.slider_n, self.slides_value[self.slider_n]

    def get_button(self):
        return self.button_n

if __name__ == '__main__':
    ui = UI_panel(com='COM16', rtplot=True)
    try:
         ui.initserial()
         ui.run()
    finally:
         ui.clean()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("user", help="User name",default='test')
#     parser.add_argument("symbol", help="Symbol drawed",default='test')
#     parser.add_argument("cnt",  help="Number of gestures intended to collect", default=60, type=int)
#     args = parser.parse_args()
#     try:
#         mask = np.array([[1] * 32 for _ in range(24)], np.uint8)
#         cnt = 0
#         mlen = 1
#         q_ft = deque(maxlen=mlen)
#         q_ar = deque(maxlen=mlen)
#         q_temp = deque(maxlen=mlen)
#         indextip1 = [0, 0]
#         indextip0_prev = [0, 0]
#         indextip1_prev = [0, 0]
#         wedge0_prev = -1
#         wedge1_prev = -1
#         bp_prev = [[0], [0]]
#         cnt0_prev = [0]
#         cnt1_prev = [0]
#         # timg = cv.imread('fingertip.jpg')
#         # print(np.shape(timg))
#         cal_cnt = 30
#         temp0_bg = [0] * 768
#         temp1_bg = [0] * 768
#         timestamp_prev = 0
#         dtime = 0
#         fullar_prev = 0
#         par_prev = 0
#         lift_flag = -1
#         hp = 5
#         yp = 5
#         x_origin = 1
#         ph_origin = (0,0)
#         x_re = 1
#         dmax = 0
#         h_correction = 0
#         y_fts = list()
#         h_fts = list()
#         vflag = 0
#         hflag = 0
#         ctemps = []
#         n_avg = 6
#
#         temp_record = list()
#         gesture_record = list()
#         record_flag = 0
#         gesture_cnt = 0
#
#         gesture_raw = list()
#         gesture_kalman = list()
#         name = args.user
#         fname = args.symbol
#         direction = list()
#         direction_kalman = list()
#         indextip0 = [0, 0]
#         xcur = 50
#         ycur = 50
#         xcur_prev = 0
#         ycur_prev = 0
#         xkalman_prev = 0
#         ykalman_prev = 0
#         while gesture_cnt < args.cnt:
#             # print(time.time())
#
#
#             # h, w = blur0.shape[:2]
#             # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
#
#             # for img in cimg
#             # dst = cv.undistort(th0, mtx, dist, None, newCameraMatrix=newcameramtx)
#             # im3.set_array(dst)
#
#             contours, hierarchy = cv.findContours(th0, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # cv2.RETR_TREE
#
#             areas = [cv.contourArea(c) for c in contours]
#             dists = list()
#
#             if len(areas) == 0:
#                 continue
#             max_index = np.argmax(areas)
#             cnt = contours[max_index]
#             hull = cv.convexHull(cnt, returnPoints=False)
#             x, y, w, h = cv.boundingRect(cnt)
#             center = [int(x + w / 2), int(y + h / 2)]
#             mask = np.zeros_like(blur0)
#             # cv.fillPoly(mask_ft, [cnt], 1)
#             defects = cv.convexityDefects(cnt, hull)
#
#             distmax = 0
#             max_df = (0,0)
#             temp1 = np.zeros_like(blur0)
#             for l in range(defects.shape[0]):
#                 s, e, fa, d = defects[l, 0]
#                 start = tuple(cnt[s][0])
#                 end = tuple(cnt[e][0])
#                 far = tuple(cnt[fa][0])
#                 cv.circle(temp1, far, 2, (127, 255, 255), -1)
#                 if d > distmax:
#                     distmax = d
#                     max_df = far
#             cv.circle(temp1, max_df, 5, (127, 255, 255), -1)
#             im3.set_array(temp1)
#             xmax = 0
#             ymin = 10000
#             xmin = 10000
#             p0 = (0, 0)
#             ifp = 0
#             ph = 0
#             iwb = 0
#             temp0 = np.zeros_like(blur0)
#             for i in range(len(hull)):
#                 cv.circle(temp0, tuple(cnt[hull[i]][0][0]), 10, (127, 255, 255), -1)
#                 if cnt[hull[i]][0][0][0] > xmax:
#                     xmax = cnt[hull[i]][0][0][0]
#                     p0 = tuple(cnt[hull[i]][0][0])
#                     ifp = i
#                 if cnt[hull[i]][0][0][1] < ymin:
#                     ymin = cnt[hull[i]][0][0][1]
#                     ph = tuple(cnt[hull[i]][0][0])
#                 if cnt[hull[i]][0][0][0] <= xmin:
#                     if cnt[hull[i]][0][0][0] == xmin:
#                         if cnt[hull[i]][0][0][1] > cnt[hull[iwb]][0][0][1]:
#                             iwb = i
#                     xmin = cnt[hull[i]][0][0][0]
#                     iwb = i-1
#                 if i < len(hull) - 1:
#                     cv.line(temp0, tuple(cnt[hull[i]][0][0]), tuple(cnt[hull[i + 1]][0][0]), [127, 255, 255], 3)
#                 else:
#                     cv.line(temp0, tuple(cnt[hull[i]][0][0]), tuple(cnt[hull[0]][0][0]), [127, 255, 255], 3)
#             pb = (0,0)
#             pwb = tuple(cnt[hull[iwb]][0][0])
#             phb = (0,0)
#             for i in range(len(hull)):
#                 phb = tuple(cnt[hull[iwb - i]][0][0])
#                 if distance(pwb, phb) > 40:
#                     break
#             slope_wp = (phb[1] - pwb[1]) / (phb[0] - pwb[0])
#
#             for i in range(len(hull)):
#                 pb = tuple(cnt[hull[ifp+i]][0][0])
#                 if distance(p0, pb) > 40:
#                     break
#             pa = (0, 0)
#             # measure finger width
#             # draw a line parallel with the finger
#             for i in range(len(hull)):
#                 pa = tuple(cnt[hull[ifp - i]][0][0])
#                 if distance(p0, pa) > 40:
#                     break
#             slope_ft_abs = (p0[1] - ph[1]) / (p0[0] - ph[0])
#             slope_ft_center = (p0[1] - center[1]) / (p0[0] - center[0])
#             slope_ft_phb = (p0[1] - phb[1]) / (p0[0] - phb[0])
#
#             cv.line(blur0, p0, pa, [127, 255, 255], 5)
#             # draw a line perpendicular to the finger
#             c0, c1, d0, d1 = getPerpCoord(p0[0], p0[1], pa[0], pa[1], 50)
#             tmp = np.zeros_like(blur0, np.uint8)
#             cv.line(tmp, (c0, c1), (d0, d1), [255, 255, 255], 5)
#             ref = np.zeros_like(blur0, np.uint8)
#             ref = cv.drawContours(ref, [cnt], -1, (255, 255, 255), 1)
#             # find intercept points of the line and the contour
#             (x_intercept, y_intercept) = np.nonzero(np.logical_and(tmp, ref))
#             # calculate distances between the points and the line
#             p1 = np.array(p0)
#             p2 = np.array(pa)
#             dists = list()
#             for m in range(len(x_intercept)):
#                 p3 = np.array([y_intercept[m], x_intercept[m]])
#                 cv.circle(blur0, tuple(p3), 2, (0, 255, 255), -1)
#                 dist = abs(np.cross(p2 - p1, p1 - p3) / np.linalg.norm(p2 - p1))
#                 if dist > 8 and dist < 40:
#                     dists.append(dist)
#             finger_width = np.nanmean(dists)
#
#             # find the point of palm fist bottom
#             for point in cnt:
#                 if point[0][0] - ph[0] < 3:
#                     phb = tuple(point[0])
#             slope_fp = (p0[1] - phb[1]) / (p0[0] - phb[0])
#             # print(slope_wp, slope_fp, slope_wp-slope_fp)
#             cv.circle(temp0, p0, 10, (127, 255, 255), -1)
#             cv.circle(temp0, pb, 10, (127, 255, 255), -1)
#             cv.circle(temp0, pwb, 10, (0, 255, 255), -1)
#             cv.circle(temp0, phb, 10, (0, 255, 255), -1)
#             rect = cv.minAreaRect(cnt)
#             box = cv.boxPoints(rect)
#             box = np.int0(box)
#
#             cv.rectangle(temp0, (x, y), (x+w, y+h), 255, 2)
#             cv.drawContours(temp0, [box], 0, (127, 0, 255), 2)
#             # cv.rectangle(temp0, (int(x-w/2), y), (int(x+w), y+h), 255, 2)
#
#             idy = 0
#             for cp in cnt:
#                 if abs(cp[0][0] - p0[0]) < 10 and cp[0][1] > p0[1]:
#                     if cp[0][1] > idy:
#                         idy = cp[0][1]
#                         indextip0 = tuple(cp[0])
#
#             th0 = th0.astype(np.bool)
#             # th0_s = morphology.skeletonize(th0)
#             # im4.set_array(th0_s)
#
#             wr = np.sum(th0.transpose()[0]) / len(th0)
#             ar_full = np.sum(np.sum(th0)) / (len(th0) * len(th0.transpose())) * 1000
#
#             # th0[indextip0[1]:] = 0
#             # for i, item in enumerate(th0):
#             #     th0[i][:center[1]] = 0
#             # th0 = np.multiply(th0, mask)
#             im2.set_array(th0)
#             ar = np.sum(np.sum(th0)) / (len(th0) * len(th0.transpose())) * 1000
#             dr = 0
#             fw = 0
#
#             if len(dists) != 0:
#                 fw = np.nanmin(dists)
#             # print(slope_fp)
#             # print('finger width:{0:.3f}, maxDefect:{1:.3f}, area ratio:{2:.3f}, full ar:{3:.3f}, wrist ratio:{4:.3f}'.format(fw, distmax,
#             #                                                                                                 ar, ar_full, wr))
#             indextip0_correct = 0
#
#             # if slope_wp - slope_fp > 1:
#             if slope_ft_abs < 0.5:
#                 if lift_flag == 0:
#                     print('LIFT, stop record')
#                     if len(temp_record) > 0:
#                         if temp_record[-1][0] - temp_record[0][0] < 1:
#                             print('bad gesture!')
#                             temp_record = list()
#                         else:
#                             gesture_record.append(temp_record)
#                             gesture_cnt += 1
#                             print('Gesture {}'.format(gesture_cnt))
#                             gesture_raw.append(direction)
#                             gesture_kalman.append(direction_kalman)
#                     temp_record = list()
#                     record_flag = 0
#                     lift_flag = 1
#                     xcur_prev = xcur
#                     ycur_prev = ycur
#                     y_fts = list()
#                     h_fts = list()
#                     vflag = 0
#                     hflag = 0
#                 elif lift_flag == -1:
#                     print('init LIFT')
#                     lift_flag = -1
#                     record_flag = 0
#
#             # elif slope_wp-slope_fp < 0.6:
#             elif slope_ft_abs > 0.7:
#                 dmax = 1
#                 for i in range(len(hull)):
#                     ptmp = np.array(cnt[hull[i]][0][0])
#                     dist = abs(np.cross(np.array(pwb) - np.array(indextip0), np.array(indextip0) - ptmp) /
#                                np.linalg.norm(np.array(pwb) - np.array(indextip0)))
#                     if dist > dmax:
#                         dmax = dist
#                         ph = tuple(ptmp)
#                 cv.circle(temp0, ph, 12, (127, 255, 255), -1)
#                 h_correction = h*abs(ph[0] - re_size[0]/2)/(re_size[0])*0.4
#                 # dmax = h + h_correction
#                 x_re = (60*216.7) / dmax
#                 y_fts.append(indextip0[0]-yp)
#                 h_fts.append(h-hp)
#
#                 # print(np.std(y_fts))
#                 # dmax = finger_width
#                 # if 10 < ph[0] and indextip0[0] < re_size[0] - 3:
#                 # if True:
#                 if lift_flag == 1 or lift_flag == -1:
#                     print(' start record')
#                     hp = h
#                     lift_flag = 0
#                     yp = indextip0[0]
#                     x_origin = x_re
#                     indextip0_correct = 0
#                     ph_origin = ph
#                     record_flag = 1
#                     f = KalmanFilter(dim_x=4, dim_z=2)
#                     f.x = np.array([[5], [5], [0], [0]])
#
#                     f.H = np.array([[1, 0, 0, 0],
#                                     [0, 1, 0, 0]])
#                     # f.G = np.array([[0, 0, 1, 0],
#                     #                 [0, 0, 0, 1]]).transpose()
#                     f.P *= 1000
#                     f.R *= 5
#                     f.Q = Q_discrete_white_noise(dim=4, dt=0.125, var=0.5)
#                     f.F = np.array([[1, 0, 0.125, 0],
#                                     [0, 1, 0, 0.125],
#                                     [0, 0, 1, 0],
#                                     [0, 0, 0, 1]])
#                     direction = list()
#                     direction_kalman = list()
#                     indextip0 = [0, 0]
#                     xcur = 50
#                     ycur = 50
#                     xcur_prev = 0
#                     ycur_prev = 0
#                     xkalman_prev = 0
#                     ykalman_prev = 0
#                 # elif lift_flag == -1:
#                 #     print('start record')
#                 #     temp_record.append([time0, temp_raw])
#                 #     lift_flag = 0
#                 #     record_flag = 1
#                 else:
#                     if record_flag == 1:
#                         temp_record.append([time0, temp_raw])
#
#                     xcur = (x_re - x_origin)
#                     # indextip0_correct = 3*(indextip0[0] - re_size[0] / 2) * (x_re - x_origin) / (x_origin)
#                     ycur = 0.5 * 0.46 * (indextip0[0] - yp)
#                     x_mov = xcur - xcur_prev
#                     y_mov = ycur - ycur_prev
#                     direction.append([np.arctan2(y_mov, x_mov) * 180 / np.pi, math.sqrt(x_mov ** 2 + y_mov ** 2)])
#                     xcur_prev = xcur
#                     ycur_prev = ycur
#                     # ycur = 20*(indextip0[0] + indextip0_correct - yp)/re_size[0] + ycur_prev
#                     f.predict()
#                     f.update([[xcur], [ycur]])
#                     x_kalman_mov = f.x[0][0] - xkalman_prev
#                     y_kalman_mov = f.x[1][0] - ykalman_prev
#                     direction_kalman.append([np.arctan2(y_kalman_mov, x_kalman_mov) * 180 / np.pi,
#                                              math.sqrt(x_kalman_mov ** 2 + y_kalman_mov ** 2)])
#                     xkalman_prev = f.x[0][0]
#                     ykalman_prev = f.x[1][0]
#                     xcur = confine(f.x[0][0], -50, 50)
#                     ycur = confine(f.x[1][0], -50, 50)
#
#             im1.set_array(temp0)
#
#             fullar_prev = ar_full
#             par_prev = ar
#             indextip0_prev = indextip0
#             # d_est = (60*216.7)/(indextip0[1]-ph[1])
#             # d_est = dr*indextip0[1] + indextip0[1]
#             # print('Lift FLAG:{0}, par:{1:.3f}, far:{2:.3f}, vx:{3:.3f}, vy:{4:.3f}, v-par:{5:.3f}, v-far:{6:.3f}, distance:{7:.3f}'.
#             #       format(lift_flag, ar, ar_full, v_x, v_lift, v_par, v_far, dr))
#
#             blur0 = cv.circle(blur0, tuple(indextip0), 10, (127, 127, 255), -1)
#             im0.set_array(blur0)
#             # 3D plot
#
#             # z_tp = map_def(indextip0[1], 0, re_size[1], 0, 10)
#             # print(x_tp, y_tp, z_tp)
#             # hl.set_data(xcur, ycur)
#             # hl.set_3d_properties(z_tp)
#             # print('Lift FLAG:{0}, x:{1:.3f},  h:{2:.3f}, y:{3:.3f}, y_correct:{4:.3f},, ycorrrect:{5:.3f}, hcorrect:{6:.3f}'
#             #       .format(lift_flag, dmax,  h, ycur, indextip0[0]+indextip0_correct, indextip0_correct, h_correction))
#             # print('Lift FLAG:{0}, slope_wp:{1:.3f},  slope_fp:{2:.3f}, delta:{3:.3f}, slope_ft_abs:{4:.3f}, slope_ft_center:{5:.3f}, slope_phb:{6:.3f}'
#             #     .format(lift_flag, slope_wp, slope_fp, slope_wp-slope_fp, slope_ft_abs, slope_ft_center, slope_ft_phb))
#             plt.pause(0.001)
