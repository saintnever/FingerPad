import serial
import matplotlib
matplotlib.use('TkAgg') # MUST BE CALLED BEFORE IMPORTING plt
from matplotlib import pyplot as plt
import queue
import threading
import animation
import seaborn as sns
import numpy as np
import time
import MLX90640

class SerialReader(threading.Thread):
    def __init__(self, stop_event, sig, serport):
        threading.Thread.__init__(self)
        self.stopped = stop_event
        self.signal = sig
        self.frame = 0
        self.n = 1667
        port = serport
        # self.s = serial.Serial(port, 9600, timeout=1, rtscts=True, dsrdtr=True)
        self.s = serial.Serial(port, 460800, timeout = 0.1, rtscts=False, dsrdtr=False)
        if not self.s.isOpen():
            self.s.open()
        print("connected: ", self.s)
        # self.s.setDTR(True)

    def run(self):
        cnt = 0
        while not self.stopped.is_set():
            # print(self.s.readline().rstrip())
            try:

                d = ord(self.s.read())
                #print(d)
                #if self.framestart:
                #    self.framestart = 0
                #self.frame.append(d)
                if d == 0x5a:
                    d = ord(self.s.read())
                    if d == 0x5a:
                        # get the frame length
                        #n = self.s.read(2)
                        #n = ((n[1] << 8) & 0xFF00) + n[0]
                        #print(n)
                        # read frame
                        self.frame = self.s.read(self.n)
                        self.signal.put(self.frame)
                        # calculate and compare CRC
                        #crc_read = self.s.read(2)
                        #print('0 is {0:x}, 1 is {1:x}'.format(crc_read[0], crc_read[1]))
                        #crc_r = ((crc_read[1] << 8) & 0xFF00) + crc_read[0]
                        #crc_cal = self.CRC(frame,n)
                        #if crc_r == crc_cal:
                        cnt += 1
                        print(cnt)
                        # print('read crc is {0:x}, cal crc is {1:x}'.format(crc_r, crc_cal))
                if d == 0xa5:
                    d = ord(self.s.read())
                    if d == 0xa5:
                    # frame = self.s.read(2*832+30)
                   
                    #print([ord(x) for x in frame])
                #    self.framestart = 1
                #    self.frame = list()
                # if dstr.find('frame'):
                #     print(dstr)
                # # print(dstr.find('Frame'))
                # if dstr.find('Frame') > 0:
                #     data_d1 = str(self.s.readline()).split(',')
                #     self.pixalarray = [float(x) for x in data_d1[1:-1]]  
                #     data_d0 = str(self.s.readline()).split(',')
                #     self.pixalarray0 = [float(x) for x in data_d0[1:-1]]
                #     if len(self.pixalarray) == 64 and len(self.pixalarray0) == 64:
                #         self.signal.put([self.pixalarray, self.pixalarray0])
            except:
                continue
        self.clean()
    
    def CRC(self, data,n):
        crc_cal = 0
        i = 0
        while i<n:
            crc_cal = (crc_cal + data[i]) & 0xFFFF
            i = i + 1
        return crc_cal
        
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

if __name__ == '__main__':
    try:
        q = queue.Queue()
        stop_event = threading.Event()
        data_reader = SerialReader(stop_event, q, 'COM11')
        data_reader.start()
        while True:
           x = 1
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
        data_reader.clean()
        data_reader.join()
