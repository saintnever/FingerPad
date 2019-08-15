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
re_size = (24*15, 32*15, 3)

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
        if data_scale < 0:
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
        if d > dmax and x * y != 0 and x!=re_size[0]-1 and y !=re_size[1]-1:
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

q0 = queue.Queue()
q1 = queue.Queue()
stop_event = threading.Event()
data_reader0 = SerialReader(stop_event, q0, 'COM15')
data_reader0.start()
data_reader1 = SerialReader(stop_event, q1, 'COM16')
data_reader1.start()

if __name__ == '__main__':
    try:
        fig, ([ax0, ax1], [ax2, ax3]) = plt.subplots(2, 2)
        # im1 = ax1.imshow(np.random.uniform(low=22, high=32, size=(20, 36)), vmin=20, vmax=36, cmap='jet')
        #                  # interpolation='lanczos')
        # im0 = ax0.imshow(np.random.uniform(low=22, high=32, size=(20, 36)), vmin=20, vmax=36, cmap='jet')
        #                  # interpolation='lanczos')
        re_size1 = (24*15, 32*15)
        im0 = ax0.imshow(np.random.uniform(low=20, high=38, size=re_size))
        im1 = ax1.imshow(np.random.uniform(low=20, high=38, size=re_size))
        im2 = ax2.imshow(np.random.uniform(low=20, high=38, size=re_size1))
        im3 = ax3.imshow(np.random.uniform(low=20, high=38, size=re_size))
        plt.tight_layout()
        plt.ion()
        fgbg = cv.createBackgroundSubtractorMOG2(history=5, detectShadows=False)
        mask = np.array([[1] * 32 for _ in range(24)], np.uint8)
        cnt = 0
        # mlen = 10
        # bg_queue = deque(maxlen=mlen)
        indextip_prev = [-1, -1]
        # timg = cv.imread('fingertip.jpg')
        # print(np.shape(timg))
        while True:
            # if cnt % 20 == 0:
            #     mask = np.array([[255] * 32 for _ in range(24)], np.uint8)
            # cnt += 1
            time0, temp0 = q0.get()
            time1, temp1 = q1.get()
            img0 = np.array([[0] * 32 for _ in range(24)], np.uint8)
            img1 = np.array([[0] * 32 for _ in range(24)], np.uint8)
            for i, x in enumerate(temp0):
                row = i // 32
                col = 31 - i % 32
                img0[int(row)][int(col)] = x
                img1[int(row)][int(col)] = temp1[i]
            img0 = image_filter(img0)
            img1 = image_filter(img1)
            # im0.set_array(rgbimg)
            # # img_raw = np.array(data0, np.uint8)
            img0 = cv.resize(img0, re_size1, interpolation=cv.INTER_LINEAR)
            img1 = cv.resize(img1, re_size1, interpolation=cv.INTER_LINEAR)
            # if np.sum(np.sum(img0, axis=1)) != 0:
            #     np.save('fingertip', img0)
            #     cv.destroyAllWindows()
            #     break
            # thinning
            # gray = cv.cvtColor(data0, cv.COLOR_BGR2GRAY)
            blur0 = cv.GaussianBlur(img0, (31, 31), 0)
            blur1 = cv.GaussianBlur(img1, (31, 31), 0)
            im0.set_array(blur0)
            im1.set_array(blur1)
            # thresh = np.max(blur)*0.9
            # ret, th = cv.threshold(blur, thresh, 255, cv.THRESH_BINARY)
            # ret, th = cv.threshold(blur0, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
            # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (17, 17))
            # # th_erode = cv.erode(th, kernel) 
            # opening = cv.morphologyEx(th, cv.MORPH_OPEN, kernel)
            # th_erode = cv.erode(opening, kernel)   
            # GRA = cv.morphologyEx(opening, cv.MORPH_GRADIENT, kernel)
            # GRA_o = cv.morphologyEx(GRA, cv.MORPH_OPEN, kernel)
            # im1.set_array(opening)
            # print(th.shape, mask.shape)
            # fgmask = fgbg.apply(opening)
            # fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
            # print(np.max(mmm))
            # im1.set_array(fgmask)

            # print(mask)
            # bg_queue.append(th)
            # if len(list(bg_queue)) == mlen:
            #     mask = np.array([[255] * 32 for _ in range(24)], np.uint8)
            #     for q in bg_queue:
            #         cv.bitwise_and(q, mask, dst=mask)
            #     bg_queue.popleft()
            # # cv.bitwise_and(th, mask, dst=mask)
            # mask_inv = cv.bitwise_not(mask)
            # cv.bitwise_and(th, mask_inv, dst=th)
            # # kernel = np.ones((2, 2), np.uint8)
            # # th = cv.erode(th, kernel)
        
            # im1.set_array(th)
            # th_thinned = thinning.skeletonize(th)
            # im2.set_array(th_erode)
            # skeleton
            # kernel = np.ones((3, 3), np.uint8)
            #
            # im0.set_array(blur)
            # # hit-or-miss detection of fingertip
            # kernel = np.array([[0, -1, 0], [0, 1, -1], [1, 1, 1]], np.uint8)
            # img_output = np.array([[0] * 32 for _ in range(24)], np.uint8)
            # hitmiss = cv.morphologyEx(th, cv.MORPH_HITMISS, kernel)
            # mmm[mmm == 1] = 255
            # indextip = trackFingertip(th_erode, indextip_prev)
            # indextip_prev = indextip
            # if indextip is not None:
            #     blur0 = cv.circle(blur0,(indextip[0], indextip[1]), 5, (0,0,255), -1)
            #     # blur[indextip[1], indextip[0]] = 125
            # im3.set_array(blur0)
            plt.pause(0.001)

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
        #     plt.pause(0.001)
        # plt.ioff()
        # plt.show()
    except KeyboardInterrupt:
        cv.imwrite('./fingertip.jpg', img0)
    finally:
        cv.destroyAllWindows()
        stop_event.set()
        data_reader0.clean()
        data_reader0.clean()
        data_reader1.join()
        data_reader1.join()
