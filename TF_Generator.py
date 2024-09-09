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
NW = [4, 8, 16, 20, 24, 32, 48, 64, 96, 128]
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
        true_if = np.squeeze(IFT)
    elif bc == 2:
        z = np.random.randint(0, N)
        x = np.concatenate((X[0][:z], X[1][z:]), axis=None)
        true_if = np.squeeze(np.concatenate((IFT[0][:z], IFT[1][z:]), axis=None))
    else:
        z = np.random.randint(0, N, size=2)
        x = np.concatenate((X[0][:min(z)], X[1][min(z):max(z)], X[2][max(z):]), axis=None)
        true_if = np.squeeze(np.concatenate((IFT[0][:min(z)], IFT[1][min(z):max(z)], IFT[2][max(z):]), axis=None))
    
    z = np.random.randint(0, N)
    ss = np.square(np.random.rand())
    vv = np.random.rand()
    x = x * np.exp(-np.square((t - t[z-1])) * np.square(ss)) + vv * (np.random.randn(N) + np.random.randn(N)) / np.sqrt(2)
    tz = t[z-1]
    stft = []

    # plots = len(NW)
    # rows = 2
    # columns = (plots + rows - 1) // rows
    # fig, axs = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))

    # kw = -1
    stfts = []
    for Nw in NW:
        # kw += 1
        STFT = np.zeros((N - Nw, NI), dtype=np.complex64)
        for k in range(Nw//2, N - Nw//2):
            x1 = x[k - Nw//2 : k + Nw//2]
            STFT[k - Nw//2, :] = np.fft.fftshift(np.fft.fft(x1, NI))
        stfts.append(STFT)

        # row_index = kw // columns
        # col_index = kw % columns
        # cax = axs[row_index, col_index].pcolor(np.abs(STFT).T, shading='auto')
        # axs[row_index, col_index].set_title(f'Nw={Nw}')
        # fig.colorbar(cax, ax=axs[row_index, col_index])

    # plt.savefig(f'tests/pc_{pc}_bc{bc}_rt{rt[0]}.pdf')
    # plt.close(fig)
    
    max_freq_bins = max(stft.shape[0] for stft in stfts)
    max_time_frames = max(stft.shape[1] for stft in stfts)
    num_stfts = len(stfts)
    unified = np.zeros((max_freq_bins, max_time_frames, num_stfts), dtype=np.complex64)

    for idx, stft in enumerate(stfts):
        pad_freq = (max_freq_bins - stft.shape[0]) // 2
        pad_time = (max_time_frames - stft.shape[1]) // 2
        freq_slice = slice(pad_freq, pad_freq + stft.shape[0])
        time_slice = slice(pad_time, pad_time + stft.shape[1])

        unified[freq_slice, time_slice, idx] = stft

    min_freq = true_if.min()
    max_freq = true_if.max()

    scaled_true_if = (true_if - min_freq) / (max_freq - min_freq) * (N-1)
    true_if_2d = np.zeros((N, N))
    for t in range(N):
        row_index = int(scaled_true_if[t])
        true_if_2d[row_index, t] = 1

    # plt.imshow(true_if_2d, cmap='hot', aspect='auto', origin='lower')
    # plt.colorbar(label='Intensity')
    # plt.title('Instantaneous Frequency Representation')
    # plt.xlabel('Time (samples)')
    # plt.ylabel('Frequency Index')
    # plt.savefig('test.pdf')
    # plt.close()

    break

