import numpy as np

def linfmgen(t):
    # zamijeniti 0.2551 sa np.random.rand()
    dt = t[1]-t[0]
    T = len(t) * dt
    wmax = 2 * np.pi * 0.5 / dt
    a2 = 4 * wmax * (0.2551 - 0.5) / 2
    d = 2 * (wmax - np.abs(a2) * T / 2)
    a1 = d * (0.2551 - 0.5)
    a0 = 2 * np.pi * (0.2551 - 0.5)
    x = np.exp(1j * a2 * np.square(t) / 2 + 1j * a1 * t + 1j * a0)
    Param = [a2, a1, a0]
    Phase = a2 * np.square(t) / 2 + a1 * t + a0
    IFT = a2 * t + a1
    return x, Param, IFT, Phase 