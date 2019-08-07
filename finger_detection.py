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
        if d > dmax:
            dmax = d
            index = [x, y]
    return index


def trackFingertip(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # cv2.RETR_TREE
    areas = [cv.contourArea(c) for c in contours]
    if areas:
        # print(areas)
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        x, y, w, h = cv.boundingRect(cnt)
        center = [x + w / 2, y + h / 2]
        ftip = find_furthest(contours[max_index], center)
    else:
        ftip = None
    return ftip

q0 = queue.Queue()
# q1 = queue.Queue()
stop_event = threading.Event()
data_reader0 = SerialReader(stop_event, q0, 'COM13')
data_reader0.start()
# data_reader1 = SerialReader(stop_event, q1, 'COM13')
# data_reader1.start()

if __name__ == '__main__':
    try:
        fig, ([ax0, ax1], [ax2, ax3]) = plt.subplots(2, 2)
        # im1 = ax1.imshow(np.random.uniform(low=22, high=32, size=(20, 36)), vmin=20, vmax=36, cmap='jet')
        #                  # interpolation='lanczos')
        # im0 = ax0.imshow(np.random.uniform(low=22, high=32, size=(20, 36)), vmin=20, vmax=36, cmap='jet')
        #                  # interpolation='lanczos')
        im0 = ax0.imshow(np.random.uniform(low=20, high=35, size=(24, 32)), cmap='jet')
        im1 = ax1.imshow(np.random.uniform(low=20, high=35, size=(24, 32)), cmap='jet')
        im2 = ax2.imshow(np.random.uniform(low=20, high=35, size=(24, 32)), cmap='jet')
        im3 = ax3.imshow(np.random.uniform(low=20, high=35, size=(24, 32)), cmap='jet')
        plt.tight_layout()
        plt.ion()
        fgbg = cv.createBackgroundSubtractorMOG2(history=5, detectShadows=False)
        mask = np.array([[1] * 32 for _ in range(24)], np.uint8)
        cnt = 0
        # mlen = 10
        # bg_queue = deque(maxlen=mlen)
        while True:
            # if cnt % 20 == 0:
            #     mask = np.array([[255] * 32 for _ in range(24)], np.uint8)
            # cnt += 1
            time0, temp0 = q0.get()
            # convert to gray image
            gray0 = colorscale(temp0, 22, 30)
            # time1, temp1 = q1.get()
            # print(time0, time1, time0-time1)
            # data0 = gray0.reshape(24, 32)
            data0 = [[0] * 32 for _ in range(24)]
            # data1 = [[0] * 32 for _ in range(24)]
            for i, x in enumerate(temp0):
                row = i // 32
                col = 31 - i % 32
                data0[int(row)][int(col)] = x
                # data1[int(row)][int(col)] = temp1[i]
            img0 = np.array(data0, np.uint8)
            # if np.sum(np.sum(img0, axis=1)) != 0:
            #     np.save('fingertip', img0)
            #     cv.destroyAllWindows()
            #     break
            # thinning
            # gray = cv.cvtColor(data0, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(img0, (5, 5), 0)
            im0.set_array(blur)
            ret, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
            opening = cv.morphologyEx(th, cv.MORPH_OPEN, kernel)
            GRA = cv.morphologyEx(opening, cv.MORPH_GRADIENT, kernel)
            # GRA_o = cv.morphologyEx(GRA, cv.MORPH_OPEN, kernel)
            # im1.set_array(opening)
            # print(th.shape, mask.shape)
            fgmask = fgbg.apply(opening)
            fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
            # print(np.max(mmm))
            im1.set_array(fgmask)

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
            # indextip = trackFingertip(th)
            # if indextip is not None:
            #     blur[indextip[1], indextip[0]] = 125
            im2.set_array(th)

            # skeleton
            # kernel = np.ones((3, 3), np.uint8)
            #
            # im0.set_array(blur)
            # # hit-or-miss detection of fingertip
            # kernel = np.array([[0, -1, 0], [0, 1, -1], [1, 1, 1]], np.uint8)
            # img_output = np.array([[0] * 32 for _ in range(24)], np.uint8)
            # erosion = cv.morphologyEx(th, cv.MORPH_HITMISS, kernel)
            # mmm[mmm == 1] = 255
            im3.set_array(GRA)
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
        # data_reader1.join()
        # data_reader1.join()
