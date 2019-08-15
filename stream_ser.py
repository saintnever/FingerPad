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

def colorscale(data, minc, maxc):
    data_scale = 256 * (data - minc) / (maxc - minc)
    if data_scale < 0:
        data_scale = 0
    elif data_scale > 255:
        data_scale = 255
    print(data, data_scale)
    return int(data_scale)


q0 = queue.Queue()
q1 = queue.Queue()
stop_event = threading.Event()
data_reader0 = SerialReader(stop_event, q0, 'COM11')
data_reader0.start()
data_reader1 = SerialReader(stop_event, q1, 'COM13')
data_reader1.start()

if __name__ == '__main__':
    try:
        fig, (ax1, ax0) = plt.subplots(1, 2)
        # im1 = ax1.imshow(np.random.uniform(low=22, high=32, size=(20, 36)), vmin=20, vmax=36, cmap='jet')
        #                  # interpolation='lanczos')
        # im0 = ax0.imshow(np.random.uniform(low=22, high=32, size=(20, 36)), vmin=20, vmax=36, cmap='jet')
        #                  # interpolation='lanczos')
        im1 = ax1.imshow(np.random.uniform(low=20, high=35, size=(24, 32)), cmap='jet')
        im0 = ax0.imshow(np.random.uniform(low=20, high=35, size=(24, 32)), cmap='jet')
        plt.tight_layout()
        plt.ion()
        while True:
            time0, temp0 = q0.get()
            time1, temp1 = q1.get()
            print(time0, time1, time0-time1)
            data0 = [[0] * 32 for _ in range(24)]
            data1 = [[0] * 32 for _ in range(24)]
            for i, x in enumerate(temp0):
                row = i // 32
                col = 31 - i % 32
                data0[int(row)][int(col)] = x
                data1[int(row)][int(col)] = temp1[i]
        #     im1.set_array(data0)
        #     im0.set_array(data1)
        #     plt.pause(0.001)
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
    finally:
        stop_event.set()
        data_reader0.clean()
        data_reader0.clean()
        data_reader1.join()
        data_reader1.join()
