import numpy as np

from hfmfmgen import hfmfmgen
from linfmgen import linfmgen
from polfmgen import polfmgen
from sinfmgen import sinfmgen

T=2
N=256

# Time vector
t = np.linspace(-T/2, T/2 - T/N, N)

wmax = 2*np.pi/2/(T/N)
NI = N

# Frequency vector
ww = np.arange(-wmax, wmax - 2*wmax/NI, 2*wmax/NI)

TRIAL = 1000

# Window sizes
NW = [4, 8, 16, 24, 32, 64, 96, 128]

# Probabilities for the number of components
PC = [0.60, 0.30, 0.10]

# Probabilities for the type of frequency modulation signals
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

    for k in range(0, bc):
        if rt[k] < TIP[0]:
            tip[k] = 1
            x, param, ift, phase = linfmgen(t)
        elif rt[k] < sum(TIP[:2]):
            tip[k] = 2
            x, param, ift, phase = sinfmgen(t)
        elif rt[k] < sum(TIP[:3]):
            tip[k] = 3
            x, param, ift, phase = polfmgen(t)
        else:
            tip[k] = 4
            x, param, ift, phase = hfmfmgen(t)
        X.append(x)
        IFT.append(ift)
        Phase.append(phase)
        Param[k] = param
    
    z=[]
    if bc == 1:
        x = X[k, :]
        true_if = IFT[:]
    elif bc == 2:
        z = np.random.randint(1, N + 1)
        x = np.concatenate((X[0, :z], X[1, z:]), axis=None)
        true_if = np.concatenate((IFT[0, :z], IFT[1, z:]), axis=None)
    else:
        z = np.random.randint(1, N+1, size=2)
        x = np.concatenate((
        X[0, :min(z)],
        X[1, min(z):max(z)],
        X[2, max(z):]), axis=None)
    
    z = np.random.randint(1, N+1)
    ss = np.random.rand()**2
    vv = np.random.rand()

    x = x * np.exp(-(t - t[z-1])**2 * ss**2) + vv * (np.random.randn(N) + np.random.randn(N)) / np.sqrt(2)
    tz = t[z-1]
    stft = []

    for kw in range(len(NW)):
        Nw = NW[kw]
        STFT = np.zeros((N - Nw, NI))
    
    for k in range(Nw//2 + 1, N - Nw//2 + 1):
        x1 = x[k - Nw//2 - 1 : k + Nw//2]
        STFT[k - Nw//2 - 1, :] = np.fft.fftshift(np.fft.fft(x1, NI))
