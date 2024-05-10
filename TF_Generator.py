import numpy as np

from hfmfmgen import hfmfmgen
from linfmgen import linfmgen
from polfmgen import polfmgen
from sinfmgen import sinfmgen

import matplotlib.pyplot as plt

T=2
N=256
t = np.linspace(-T/2, T/2 - T/N, N)
wmax = 2*np.pi/2/(T/N)
NI = N
ww = np.arange(-wmax, wmax - 2*wmax/NI, 2*wmax/NI)
TRIAL = 1000
NW = [4, 8, 16, 24, 32, 64, 96, 128]
PC = [0.60, 0.30, 0.10]
TIP = [0.10, 0.30, 0.30, 0.30]

for trial in range(0, TRIAL):
    pc = np.random.rand()
    if pc < PC[0]: 
        bc = 1
    elif pc < PC[0] + PC[1]:
        bc = 2
    else:
        bc = 3

    rt = np.random.rand(bc)
    X = []
    Param = {}
    IFT = []
    Phase = []
    tip = []

    for k in range(bc):
        if rt[k] < TIP[0]:
            tip.append(1)
            x, param, ift, phase = linfmgen(t)
        elif rt[k] < sum(TIP[:2]):
            tip.append(2)
            x, param, ift, phase = sinfmgen(t)
        elif rt[k] < sum(TIP[:3]):
            tip.append(3)
            x, param, ift, phase = polfmgen(t)
        else:
            tip.append(4)
            x, param, ift, phase = hfmfmgen(t)

        X.append(x)
        IFT.append(ift)
        Phase.append(phase)
        Param[k] = param
    
    if bc == 1:
        x = X[k]
        true_if = IFT
    elif bc == 2:
        z = np.random.randint(0, N)
        x = np.concatenate((X[0][:z], X[1][z:]), axis=None)
        true_if = np.concatenate((IFT[0][:z], IFT[1][z:]), axis=None)
    else:
        z = np.random.randint(0, N, size=2)
        x = np.concatenate((X[0][:min(z)], X[1][min(z):max(z)], X[2][max(z):]), axis=None)
        true_if = np.concatenate((IFT[0][:min(z)], IFT[1][min(z):max(z)], IFT[2][max(z):]), axis=None)
    
    z = np.random.randint(0, N)
    ss = np.square(np.random.rand())
    vv = np.random.rand()
    x = x * np.exp(-np.square((t - t[z-1])) * np.square(ss)) + vv * (np.random.randn(N) + np.random.randn(N)) / np.sqrt(2)
    tz = t[z-1]
    stft = []

    kw = -1
    for Nw in NW:
        kw+=1
        STFT = np.zeros((N - Nw, NI), dtype=np.complex64)
        for k in range(Nw//2, N - Nw//2):
            x1 = x[k - Nw//2 : k + Nw//2]
            STFT[k - Nw//2, :] = np.fft.fftshift(np.fft.fft(x1, NI))
    
    #     m = np.max(np.abs(STFT), axis=0)  # Maximum values
    #     a = np.argmax(np.abs(STFT), axis=0)  # Indices of maximum values

    #     fig, axs = plt.subplots(2, 8)  # Create a 2x8 subplot grid
    #     cax = axs[kw//8, kw%8].pcolor(np.abs(STFT).T, shading='auto')  # Transpose STFT for correct orientation
    #     fig.colorbar(cax, ax=axs[kw//8, kw%8])  # Add a colorbar

    #     plt.show()

    # input("Press Enter to continue...")

