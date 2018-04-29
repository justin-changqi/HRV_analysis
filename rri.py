import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import cmath

class RRI:
    def __init__(self, file_name):
        df = pd.read_csv(file_name)
        self.rri_data = np.array(df['rri'].tolist())
        # print (self.rri_data)

    def plotData(self, y, title):
        plt.figure(title)
        plt.plot(y, 'r-')
        plt.ylabel('RRI')
        pass

    def plotSpectrum(self, x, y, title):
        plt.figure(title)
        for i in range(len(y)):
            if y[i] == 0:
                x[i] = 0
        plt.plot(y, x, 'r-')
        plt.ylabel('Enery')

    def getSDNN(self, data):
        # total heart beats
        THB = len(data)
        # mean of R-R interval
        MRR = np.mean(data[1:])
        # standard deviation of normal to normal RR intervals
        SDNN = 0
        for i in range(1, THB):
            SDNN += math.pow(data[i] - MRR, 2)
        return math.sqrt(SDNN / (THB-1))

    def getRMSSD(self, data):
        # total heart beats
        THB = len(data)
        # root mean square of successive NN interval differences
        RMSSD = 0.
        for i in range(2, THB):
            RMSSD += math.pow(data[i]-data[i-1], 2)
        return math.sqrt(RMSSD / (THB-2))

    def getFFT(self,data):
        sp = np.fft.fft(data)
        freq = np.fft.fftfreq(np.arange(len(data)).shape[-1])
        freq_sorted_indexes = np.argsort(freq)
        sorted_sp = []
        sorted_freq = []
        for index in freq_sorted_indexes:
            sorted_sp.append(cmath.polar(sp[index])[0])
            sorted_freq.append(freq[index])
        return np.array(sorted_sp), np.array(sorted_freq)

    def getHFLF(self, sp, freq):
        HF = 0
        LF = 0
        for i in range(len(freq)):
            if freq[i] >= 0.15 and freq[i] < 0.4:
                # HF
                HF += sp[i]
            if freq[i] >= 0.04 and freq[i] < 0.15:
                # LF
                LF += sp[i]
        return HF, LF

    def BergerResampling(self, data, freq):
        nornal_scalar = math.pow(np.mean(data), 2)
        pi = math.pi
        N = len(data)
        tn = sum(data)
        Pc_f = []
        for i in range(len(freq)):
            f = freq[i]
            if (f==0): f = 1
            w = 2*pi*f*tn
            sigma_cos = 0
            sigma_sin = 0
            for k in range(len(data)):
                tk = sum(data[:k])
                sigma_cos += math.cos(2*pi*f*tk)
                sigma_sin += math.sin(2*pi*f*tk)
            s0 = ((N*math.sin(w))/w) - sigma_cos
            s1 = ((N*(math.cos(w)-1)/w)) + sigma_sin
            Pc = tn/(N**2)*(math.pow(s0, 2) + math.pow(s1, 2))
            Pc_f.append(Pc/nornal_scalar)
        return np.array(Pc_f), freq


if __name__ == "__main__":
    rri = RRI('RRI.csv')
    # rri.plotData(rri.rri_data, 'RRI')
    SDNN = rri.getSDNN(rri.rri_data)
    RMSSD = rri.getRMSSD(rri.rri_data)
    print ('SDNN:', '%.6f' %SDNN, 's')
    print ('RMSSD:', '%.6f' %RMSSD, 's')
    # FFT
    sp_rri, freq_rri = rri.getFFT(rri.rri_data)
    HF, LF = rri.getHFLF(sp_rri, freq_rri)
    rri.plotSpectrum(sp_rri, freq_rri, 'Spectrum')
    # Berger Resampling
    sp_br, freq_br = rri.BergerResampling(rri.rri_data, freq_rri)
    HF_br, LF_br = rri.getHFLF(sp_br, freq_br)
    rri.plotSpectrum(sp_br, freq_br, 'Berger Spectrum')
    print ('=========== FFT ===========')
    print (' HF:\t\t', '%.3f' %HF)
    print (' LF:\t\t', '%.3f' %LF)
    print (' LF/HF:\t\t', '%.3f' %(LF/HF))
    print ('==== Berger Resampling ====')
    print (' HF:\t\t', '%.3f' %HF_br)
    print (' LF:\t\t', '%.3f' %LF_br)
    print (' LF/HF:\t\t', '%.3f' %(LF_br/HF_br))
    plt.show()
