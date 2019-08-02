import time
import board
from digitalio import DigitalInOut, Direction, Pull
import busio
import math
from adafruit_bus_device.i2c_device import I2CDevice

debug = 1
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
print('SCL is {}, SDA is {}'.format(board.SCL, board.SDA))

class MLX90640:
    def __init__(self):
        self.device = I2CDevice(i2c, 0x33)
        self.eeData = self.getReg(0x2400, 832)
        self.kVdd, self.vdd25 = self.ExtractVDDParameters()
        self.KvPTAT,self.KtPTAT, self.vPTAT25, self.alphaPTAT = self.ExtractPTATParameters()
        self.gainEE = self.eeData[48]-65536 if self.eeData[48] > 32767 else self.eeData[48]
        self.tgc = (self.eeData[60] & 0x00FF) - 256 if (self.eeData[60] & 0xFF) > 127 else self.eeData[60] & 0xFF
        self.cpKv, self.cpKta, self.cpAlpha, self.cpOffset = self.ExtractCPParameters()
        self.resolutionEE = (self.eeData[56] & 0x3000) >> 12
        self.calibrationModeEE, self.ilChessC = self.ExtractCILCParameters()
        self.KsTa = (((self.eeData[60] & 0xFF00)>>8) - 256)/8192.0 if (self.eeData[60] & 0xFF00)>>8 > 127 else ((self.eeData[60] & 0xFF00)>>8)/8192.0
        self.ct, self.ksTo = self.ExtractKsToParameters()
        self.alpha = self.ExtractAlphaParameters()
        self.offset = self.ExtractOffsetParameters()  
        self.kta = self.ExtractKtaPixelParameters()   
        self.kv = self.ExtractKvPixelParameters()
        # print(self.offset, self.kta, self.kv)
        self.emissivity = 0.95
        self.TA_SHIFT = 8
        # self.brokenPixels[5]
        # self.outlierPixels[5]  

    def getRego(self, addr, num=1):
        temp = bytearray(2)
        out = [0xFFFF for _ in range(num)]
        with self.device:
            i = 0
            while i < num:
                self.device.write_then_readinto(bytes([(addr + i) >> 8, (addr + i) & 0xFF]), temp, stop=False)
                # self.device.readinto(temp)
                out[i] = (temp[0]<<8) + temp[1]
                i = i + 1
        if num == 1:
            return out[0]
        else:
            return out

    def getReg(self, addr, num=1):
        ts = time.monotonic()
        temp = bytearray(num * 2)
        out = [0xFFFF for _ in range(num)]
        t1 = time.monotonic()
        with self.device:
            self.device.write_then_readinto(bytes([addr >> 8, addr & 0xFF]), temp, stop=False)
        t2 = time.monotonic()
        for i in range(num):
            out[i] = (temp[2 * i] << 8) + temp[2 * i + 1]
        t3 = time.monotonic()
        if num > 700:
            print('getReg time, init {}, actual read {}, assemble {}'.format(t1-ts, t2-t1, t3-t2))
        if num == 1:
            return out[0]
        else:
            return out

    def getReg_raw(self, addr, values, num=1):
        with self.device:
            self.device.write_then_readinto(bytes([addr >> 8, addr & 0xFF]), values, stop=False)
    
    def assemble(self, values, out):
        num = int(len(values)/2)
        # out = [0xFFFF for _ in range(num)]
        for i in range(num):
            out[i] = (values[2 * i] << 8) + values[2 * i + 1]
        # return out

    def setReg(self, addr, value, num=1):
        if num == 1:
            value = [value]
        elif len(value) != num:
            return -1
        with self.device:
            i = 0
            while i < num:
                self.device.write(bytes([(addr + i) >> 8, (addr + i) & 0xFF, value[i] >> 8, value[i] & 0xFF]), stop=False)
                # device.write(bytes([value[i] >> 8, value[i] & 0xFF]), stop=False)
                i = i + 1
        return 0

    def i2cClock(self, freq):
        i2c.deinit()
        i2c_hf = busio.I2C(board.SCL, board.SDA, frequency=freq)
        self.device = I2CDevice(i2c_hf, 0x33)

    def ExtractVDDParameters(self):
        kVdd = self.eeData[51]
        kVdd = (self.eeData[51] & 0xFF00) >> 8
        if(kVdd > 127):
            kVdd = kVdd - 256
        kVdd = 32 * kVdd
        vdd25 = self.eeData[51] & 0x00FF
        vdd25 = ((vdd25 - 256) << 5) - 8192
        return kVdd, vdd25

    def ExtractPTATParameters(self):
        KvPTAT = (self.eeData[50] & 0xFC00) >> 10
        if(KvPTAT > 31):
            KvPTAT = KvPTAT - 64
        KvPTAT = KvPTAT/4096
        
        KtPTAT = self.eeData[50] & 0x03FF
        if(KtPTAT > 511):
            KtPTAT = KtPTAT - 1024
        KtPTAT = KtPTAT/8
        
        vPTAT25 = self.eeData[49]
        
        alphaPTAT = (self.eeData[16] & 0xF000) / pow(2, 14.0) + 8.0
        
        return KvPTAT, KtPTAT, vPTAT25, alphaPTAT

    def ExtractCPParameters(self):
        alphaSP = [0] * 2
        offsetSP = [0] * 2
        alphaScale = ((self.eeData[32] & 0xF000) >> 12) + 27
        offsetSP[0] = (self.eeData[58] & 0x03FF)
        if offsetSP[0] > 511:
            offsetSP[0] = offsetSP[0] - 1024
        offsetSP[1] = (self.eeData[58] & 0xFC00) >> 10
        if offsetSP[1] > 31:
            offsetSP[1] = offsetSP[1] - 64
        offsetSP[1] = offsetSP[1] + offsetSP[0]
        alphaSP[0] = (self.eeData[57] & 0x03FF)
        if alphaSP[0] > 511:
            alphaSP[0] = alphaSP[0] - 1024
        alphaSP[0] = alphaSP[0] /  pow(2,alphaScale)
        alphaSP[1] = (self.eeData[57] & 0xFC00) >> 10
        if alphaSP[1] > 31:
            alphaSP[1] = alphaSP[1] - 64
        alphaSP[1] = (1 + alphaSP[1]/128) * alphaSP[0]
        
        cpKta = (self.eeData[59] & 0x00FF)
        if cpKta > 127:
            cpKta = cpKta - 256
        ktaScale1 = ((self.eeData[56] & 0x00F0) >> 4) + 8    
        cpKta = cpKta / float(pow(2,ktaScale1))
        
        cpKv = (self.eeData[59] & 0xFF00) >> 8
        if cpKv > 127:
            cpKv = cpKv - 256
        kvScale = (self.eeData[56] & 0x0F00) >> 8
        cpKv = cpKv / float(pow(2, kvScale))

        return cpKv, cpKta, alphaSP, offsetSP

    def ExtractCILCParameters(self):
        ilChessC = [0]*3
        calibrationModeEE = (self.eeData[10] & 0x0800) >> 4
        calibrationModeEE = calibrationModeEE ^ 0x80
        ilChessC[0] = (self.eeData[53] & 0x003F)
        if ilChessC[0] > 31:
            ilChessC[0] = ilChessC[0] - 64
        ilChessC[0] = ilChessC[0] / 16.0
        ilChessC[1] = (self.eeData[53] & 0x07C0) >> 6
        if ilChessC[1] > 15:
            ilChessC[1] = ilChessC[1] - 32
        ilChessC[1] = ilChessC[1] / 2.0
        ilChessC[2] = (self.eeData[53] & 0xF800) >> 11
        if ilChessC[2] > 15:
            ilChessC[2] = ilChessC[2] - 32
        ilChessC[2] = ilChessC[2] / 8.0
        return calibrationModeEE, ilChessC
    
    def ExtractKsToParameters(self):
        ct = ksTo = [0]*4
        step = ((self.eeData[63] & 0x3000) >> 12) * 10
        ct[0] = -40
        ct[1] = 0
        ct[2] = (self.eeData[63] & 0x00F0) >> 4
        ct[3] = (self.eeData[63] & 0x0F00) >> 8
        ct[2] = ct[2]*step
        ct[3] = ct[2] + ct[3]*step
        KsToScale = (self.eeData[63] & 0x000F) + 8
        KsToScale = 1 << KsToScale
        ksTo[0] = self.eeData[61] & 0x00FF
        ksTo[1] = (self.eeData[61] & 0xFF00) >> 8
        ksTo[2] = self.eeData[62] & 0x00FF
        ksTo[3] = (self.eeData[62] & 0xFF00) >> 8
        for i in range(4):
            if ksTo[i] > 127:
                ksTo[i] = ksTo[i] -256
            ksTo[i] = ksTo[i] / KsToScale
        return ct, ksTo

    def ExtractAlphaParameters(self):
        accRow = [0]*24
        accColumn = [0] * 32
        alpha = [0] * 768
        p = 0
        accRemScale = self.eeData[32] & 0x000F
        accColumnScale = (self.eeData[32] & 0x00F0) >> 4
        accRowScale = (self.eeData[32] & 0x0F00) >> 8
        alphaScale = ((self.eeData[32] & 0xF000) >> 12) + 30
        alphaRef = self.eeData[33]
        
        for i in range(6):
            p = i * 4
            accRow[p + 0] = (self.eeData[34 + i] & 0x000F)
            accRow[p + 1] = (self.eeData[34 + i] & 0x00F0) >> 4
            accRow[p + 2] = (self.eeData[34 + i] & 0x0F00) >> 8
            accRow[p + 3] = (self.eeData[34 + i] & 0xF000) >> 12
        
        for i in range(24):
            if accRow[i] > 7:
                accRow[i] = accRow[i] - 16
        
        for i in range(8):
            p = i * 4
            accColumn[p + 0] = (self.eeData[40 + i] & 0x000F)
            accColumn[p + 1] = (self.eeData[40 + i] & 0x00F0) >> 4
            accColumn[p + 2] = (self.eeData[40 + i] & 0x0F00) >> 8
            accColumn[p + 3] = (self.eeData[40 + i] & 0xF000) >> 12
        
        for i in range(32):
            if accColumn[i] > 7:
                accColumn[i] = accColumn[i] - 16
            
        for i in range(24):
            for j in range(32):
                p = 32 * i +j
                alpha[p] = (self.eeData[64 + p] & 0x03F0) >> 4
                if alpha[p] > 31:
                    alpha[p] = alpha[p] - 64
                alpha[p] = alpha[p]*(1 << accRemScale)
                alpha[p] = (alphaRef + (accRow[i] << accRowScale) + (accColumn[j] << accColumnScale) + alpha[p])
                alpha[p] = alpha[p] / float(pow(2, alphaScale))
        return alpha

    def ExtractOffsetParameters(self):
        occRow = [0]*24
        occColumn = [0] * 32
        offset = [0] * 768
        p = 0
        occRemScale = self.eeData[16] & 0x000F
        occColumnScale = (self.eeData[16] & 0x00F0) >> 4
        occRowScale = (self.eeData[16] & 0x0F00) >> 8
        offsetRef = self.eeData[17]
        if offsetRef > 32767:
            offsetRef -= 65536
        
        for i in range(6):
            p = i * 4
            occRow[p + 0] = (self.eeData[18 + i] & 0x000F)
            occRow[p + 1] = (self.eeData[18 + i] & 0x00F0) >> 4
            occRow[p + 2] = (self.eeData[18 + i] & 0x0F00) >> 8
            occRow[p + 3] = (self.eeData[18 + i] & 0xF000) >> 12
        
        for i in range(24):
            if occRow[i] > 7:
                occRow[i] = occRow[i] - 16
        
        for i in range(8):
            p = i * 4
            occColumn[p + 0] = (self.eeData[24 + i] & 0x000F)
            occColumn[p + 1] = (self.eeData[24 + i] & 0x00F0) >> 4
            occColumn[p + 2] = (self.eeData[24 + i] & 0x0F00) >> 8
            occColumn[p + 3] = (self.eeData[24 + i] & 0xF000) >> 12
        
        for i in range(32):
            if occColumn[i] > 7:
                occColumn[i] = occColumn[i] - 16
            
        for i in range(24):
            for j in range(32):
                p = 32 * i +j
                offset[p] = (self.eeData[64 + p] & 0xFC00) >> 10
                if offset[p] > 31:
                    offset[p] = offset[p] - 64
                offset[p] = offset[p]*(1 << occRemScale)
                offset[p] = (offsetRef + (occRow[i] << occRowScale) + (occColumn[j] << occColumnScale) + offset[p])
        return offset

    def ExtractKtaPixelParameters(self):
        p = 0
        KtaRC = [0] * 4
        kta = [0] * 768

        KtaRoCo = (self.eeData[54] & 0xFF00) >> 8
        if KtaRoCo > 127:
            KtaRoCo = KtaRoCo - 256
        KtaRC[0] = KtaRoCo
        
        KtaReCo = (self.eeData[54] & 0x00FF)
        if (KtaReCo > 127):
            KtaReCo = KtaReCo - 256
        KtaRC[2] = KtaReCo
        
        KtaRoCe = (self.eeData[55] & 0xFF00) >> 8
        if (KtaRoCe > 127):
            KtaRoCe = KtaRoCe - 256
        KtaRC[1] = KtaRoCe
        
        KtaReCe = (self.eeData[55] & 0x00FF)
        if (KtaReCe > 127):
            KtaReCe = KtaReCe - 256
        KtaRC[3] = KtaReCe
    
        ktaScale1 = ((self.eeData[56] & 0x00F0) >> 4) + 8
        ktaScale2 = (self.eeData[56] & 0x000F)

        for i in range(24):
            for j in range(32):
                p = 32 * i +j
                split = int(2*(p/32 - (p/64)*2) + p%2)
                kta[p] = (self.eeData[64 + p] & 0x000E) >> 1
                if (kta[p] > 3):
                    kta[p] = kta[p] - 8
                kta[p] = kta[p] * (1 << ktaScale2)
                kta[p] = KtaRC[split] + kta[p]
                kta[p] = kta[p] / float(pow(2, ktaScale1))
        return kta

    def ExtractKvPixelParameters(self):
        p = 0
        KvT = [0] * 4
        kv = [0] * 768

        KvRoCo = (self.eeData[52] & 0xF000) >> 12
        if (KvRoCo > 7):
            KvRoCo = KvRoCo - 16
        KvT[0] = KvRoCo
        
        KvReCo = (self.eeData[52] & 0x0F00) >> 8
        if (KvReCo > 7):
            KvReCo = KvReCo - 16
        KvT[2] = KvReCo
        
        KvRoCe = (self.eeData[52] & 0x00F0) >> 4
        if (KvRoCe > 7):
            KvRoCe = KvRoCe - 16
        KvT[1] = KvRoCe
        
        KvReCe = (self.eeData[52] & 0x000F)
        if (KvReCe > 7):
            KvReCe = KvReCe - 16
        KvT[3] = KvReCe
    
        kvScale = (self.eeData[56] & 0x0F00) >> 8

        for i in range(24):
            for j in range(32):
                p = 32 * i +j
                split = int(2*(p/32 - (p/64)*2) + p%2)
                kv[p] = KvT[split]
                kv[p] = kv[p] / pow(2, kvScale)
        return kv

    def getVDD(self, frameData):
        vdd = frameData[810]
        if vdd > 32767:
            vdd -= 65536
        resRAM = (frameData[832] & 0x0C00) >> 10
        resCorr = pow(2, self.resolutionEE) / pow(2, resRAM)
        vdd = (resCorr * vdd - self.vdd25) / self.kVdd + 3.3
        return vdd

    def getTa(self, frameData):
        vdd = self.getVDD(frameData)
        ptat = frameData[800]
        if ptat > 32767:
            ptat -= 65536
        ptatArt = frameData[768]
        if ptatArt > 32767:
            ptatArt -= 65536
        ptatArt = (ptat / (ptat * self.alphaPTAT + ptatArt)) * pow(2, 18.0)
        ta = (ptatArt / (1 + self.KvPTAT * (vdd - 3.3)) - self.vPTAT25)
        ta = ta / self.KtPTAT + 25        
        return ta

    def setRefreshRate(self, refreshRate):
        value = (refreshRate & 0x07) << 7
        controlReg = self.getReg(0x800D)
        value = (controlReg & 0xFC7F) | value
        error = self.setReg(0x800D, value)
        # print('Set refresh rate from 0x{0:x} to 0x{1:x}'.format((controlReg>>7)&0x07, self.getReg(0x800D)))
        return error

    def setSubpageMode(self, mode=True):
        controlReg = self.getReg(0x800D)
        if mode:
            value = controlReg | 0x0001
        else:
            value = controlReg & 0xFFFE
        error = self.setReg(0x800D, value)
        # print('Set subpage mode from 0x{0:x} to 0x{1:x}'.format(controlReg&0x0001, self.getReg(0x800D)&0x0001))
        return error

    def getFrameDate(self, values, frame_data):
        dataReady = 1
        controlReg = statusReg = error = 1
        cnt = 0
        #  Wait for new data to be available
        dataReady = 0
        now = time.monotonic()
        while dataReady == 0:
            statusReg = self.getReg(0x8000)
            dataReady = statusReg & 0x0008
        t1 = time.monotonic()
        while dataReady != 0 and cnt < 5:
            # Set bits 4 and 5: Start measurement and enable overwrite
            error = self.setReg(0x8000, 0x0030)
            if (error == -1):
                return error
            #Read 832 pixels
            subpage = self.getReg(0x8000) & 0x0001
            # if subpage == 1:
            self.getReg_raw(0x0400, values, 832)
            statusReg = self.getReg(0x8000)
            dataReady = statusReg & 0x0008
            # print('0x{0:x}, dataReady is {1:x}'.format(statusReg, dataReady))
            cnt += 1
        if cnt > 4:
            return - 8
        t2 = time.monotonic()
        if debug:
            print('in getFrame, initial wait is {}, actual read time is {}, total time is {}'.format(t1 - now, t2 - t1, t2 - now))
            # print('control reg is 0x{0:x}, statusReg is 0x{1:x}'.format(frame_data[-2], frame_data[-1]))
        return statusReg & 0x0001

        # return frame_data

    def CalculateTo(self, frameData, result):
        try:
            if len(frameData) != 834:
                return - 1
        except:
            return -1
        # if frameData < 0:
        #     return -1
        irDataCP = [0]* 2
        alphaCorrR = [0] * 4
        # result = [0] * 768
        subPage = frameData[833]
        vdd = self.getVDD(frameData)
        ta = self.getTa(frameData)
        tr = ta - self.TA_SHIFT
        ta4 = pow((ta + 273.15), 4.0)
        tr4 = pow((tr + 273.15), 4.0)
        taTr = tr4 - (tr4 - ta4) / self.emissivity
        
        alphaCorrR[0] = 1 / (1 + self.ksTo[0] * 40)
        alphaCorrR[1] = 1 
        alphaCorrR[2] = (1 + self.ksTo[2] * self.ct[2])
        alphaCorrR[3] = alphaCorrR[2] * (1 + self.ksTo[3] * (self.ct[3] - self.ct[2]))
        
        #------------------------- Gain calculation -----------------------------------    
        gain = frameData[778]
        if(gain > 32767):
            gain = gain - 65536
        
        gain = self.gainEE / gain 
    
        #------------------------- To calculation -------------------------------------    
        mode = (frameData[832] & 0x1000) >> 5
        
        irDataCP[0] = frameData[776]  
        irDataCP[1] = frameData[808]
        for i in range(2):
            if(irDataCP[i] > 32767):
                irDataCP[i] = irDataCP[i] - 65536
            irDataCP[i] = irDataCP[i] * gain
        irDataCP[0] = irDataCP[0] - self.cpOffset[0] * (1 + self.cpKta * (ta - 25)) * (1 + self.cpKv * (vdd - 3.3))
        if( mode ==  self.calibrationModeEE):
            irDataCP[1] = irDataCP[1] - self.cpOffset[1] * (1 + self.cpKta * (ta - 25)) * (1 + self.cpKv * (vdd - 3.3))
        else:
            irDataCP[1] = irDataCP[1] - (self.cpOffset[1] + self.ilChessC[0]) * (1 + self.cpKta * (ta - 25)) * (1 + self.cpKv * (vdd - 3.3))    
        # print('subpage {}, gain {}, cpKta {}, cpkv {}, mode{}, modeEE {}, vdd{}'.format(subPage, gain, self.cpKta, self.cpKv, mode, self.calibrationModeEE, vdd))

        for pixelNumber in range(768):
            ilPattern = int(pixelNumber / 32 - (pixelNumber / 64) * 2)
            chessPattern = ilPattern ^ int(pixelNumber - (pixelNumber/2)*2) 
            conversionPattern = ((pixelNumber + 2) / 4 - (pixelNumber + 3) / 4 + (pixelNumber + 1) / 4 - pixelNumber / 4) * (1 - 2 * ilPattern)
            pattern = ilPattern if mode == 0 else chessPattern
            # if debug:
            #     print(pattern, ilPattern, chessPattern, subPage)
            if(pattern == frameData[833]):    
                irData = frameData[pixelNumber] 
                if(irData > 32767):
                    irData = irData - 65536
                irData = irData * gain
                irData = irData - self.offset[pixelNumber]*(1 + self.kta[pixelNumber]*(ta - 25))*(1 + self.kv[pixelNumber]*(vdd - 3.3))
                if(mode !=  self.calibrationModeEE):
                    irData = irData + self.ilChessC[2] * (2 * ilPattern - 1) - self.ilChessC[1] * conversionPattern 
                
                irData = irData / self.emissivity
        
                irData = irData - self.tgc * irDataCP[subPage]
                
                alphaCompensated = (self.alpha[pixelNumber] - self.tgc * self.cpAlpha[subPage])*(1 + self.KsTa * (ta - 25.0))
                
                Sx = pow(alphaCompensated, 3.0) * (irData + alphaCompensated * taTr)
                Sx = math.sqrt(math.sqrt(Sx)) * self.ksTo[1]
                
                To = math.sqrt(math.sqrt(irData/(alphaCompensated * (1 - self.ksTo[1] * 273.15) + Sx) + taTr)) - 273.15
                        
                if(To < self.ct[1]):
                    ranget = 0
                elif(To < self.ct[2]):
                    ranget = 1            
                elif(To < self.ct[3]):
                    ranget = 2            
                else:
                    ranget = 3              
                
                To = math.sqrt(math.sqrt(irData / (alphaCompensated * alphaCorrR[ranget] * (1 + self.ksTo[ranget] * (To - self.ct[ranget]))) + taTr)) - 273.15
                
                result[pixelNumber] = To
        # return result

if __name__ == '__main__':
    sensor = MLX90640()
    # addr = sensor.getReg(0x240F)
    # print('device address is 0x{0:x}, old read is 0x{1:x}'.format(addr, sensor.getReg(0x240F)))
    # print(sensor.kVdd)
    error = sensor.setRefreshRate(0x04)
    # print(error)
    error = sensor.setSubpageMode(True)
    sensor.i2cClock(1000000)
    # print(error)
    # sensor.i2cClock(1000000)
    # led = DigitalInOut(board.P0_13)
    # led.direction = Direction.OUTPUT
    values = bytearray(832* 2)
    frame_data = [0xFFFF for _ in range(834)]
    frameTo = [0] * 768
    psubpage = 0
    while True:
        now = time.monotonic()
        subpage = sensor.getFrameDate(values, frame_data)
        if subpage != psubpage:
            sensor.assemble(values, frame_data)
            frame_data[-2] = sensor.getReg(0x800D)
            frame_data[-1] = subpage
            psubpage = subpage
            t1 = time.monotonic()
            if subpage == 0:
                sensor.CalculateTo(frame_data, frameTo)
                t2 = time.monotonic()
                if debug:
                    print('total time is  {}, getFrame time is {}, calculate Frame time is {}'.format(t2-now, t1-now, t2-t1))
                    print(','.join('{:.2f}'.format(x) for x in frameTo))

            
    
