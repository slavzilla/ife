import numpy as np

from data_processing import calculate_spectrograms, map_if_to_2d_image, modulate_and_add_noise, normalize_and_map, unify_spectrograms
from dataset import generate_tfrecord_filename, write_tfrecord
from hfmfmgen import hfmfmgen
from linfmgen import linfmgen
from plotting import plot, plot_instantaneous_frequency, plot_stfts
from polfmgen import polfmgen
from sinfmgen import sinfmgen


T=2
N=256
t = np.linspace(-T/2, T/2 - T/N, N)
wmax = 2*np.pi/2/(T/N)
NI = N
ww = np.arange(-wmax, wmax - 2*wmax/NI, 2*wmax/NI)
TRIAL = 10000
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
    
    x = modulate_and_add_noise(x, t)

    #tz = t[z-1]

    specs = calculate_spectrograms(x, NW, N, NI)
    
    unified_specs = unify_spectrograms(specs)

    #plot_stfts(specs, NW)

    scaled_true_if = normalize_and_map(true_if, N-1)
    true_if_2d = map_if_to_2d_image(scaled_true_if, N)

    #plot_instantaneous_frequency(true_if_2d)

    filename = generate_tfrecord_filename(trial)

    write_tfrecord(filename, unified_specs, true_if_2d)

    print('Successfully created tfrecord:' + filename)

