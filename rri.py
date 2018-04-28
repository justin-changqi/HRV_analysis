import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

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
            sorted_sp.append(sp[index])
            sorted_freq.append(freq[index])
        return np.array(sorted_sp), np.array(sorted_freq)


if __name__ == "__main__":
    rri = RRI('RRI.csv')
    # rri.plotData(rri.rri_data, 'RRI')
    SDNN = rri.getSDNN(rri.rri_data)
    RMSSD = rri.getRMSSD(rri.rri_data)
    sp_rri, freq_rri = rri.getFFT(rri.rri_data)
    rri.plotSpectrum(sp_rri, freq_rri, 'Spectrum')
    print ('SDNN: ', '%.6f' %SDNN, 's')
    print ('RMSSD: ', '%.6f' %RMSSD, 's')
    plt.show()
