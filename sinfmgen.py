import numpy as np

def sinfmgen(t):
    # zamijeniti 0.2551 sa np.random.rand()
    dt = t[1]-t[0]
    wmax = 2 * np.pi * 0.5 / dt
    be = 8 * np.pi * (0.2551 - 0.5)
    gm = 2 * np.pi * (0.2551 - 0.5)
    al = 2 * wmax / be * (0.2551 - 0.5)
    d = 2 * (wmax - np.abs(al * be))
    a1 = d * (0.2551 - 0.5)
    a0 = 2 * np.pi * (0.2551 - 0.5)
    x = np.exp(1j * al * np.sin(be *t + gm) + 1j * a1 * t + 1j * a0)
    Param = [al, be, gm, a1, a0]
    Phase = al * np.sin(be * t + gm) + a1 * t + a0
    IFT = al * be * np.cos(be * t + gm) + a1
    return x, Param, IFT, Phase
