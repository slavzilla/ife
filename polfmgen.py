import numpy as np

def polfmgen(t):
    dt = t[1]-t[0]
    wmax = 2 * np.pi * 0.5 / dt
    cm = 64 * np.pi
    c = []
    r = 1
    C = []
    mif = 0
    while r == 1 or (r <= 20 and mif <= wmax):
        c += [cm * (np.random.rand() - 0.5)]
        descending_sequence = np.arange(r-1, 0, -1)
        coefficients = c[:r-1] * descending_sequence
        ift = np.polyval(coefficients, t)
        mif = np.max([np.abs(np.min(ift)), np.abs(np.max(ift))])
        if mif <= wmax:
            r = r + 1
            C = c.copy()
    pht = np.polyval(C, t)
    x = np.exp(1j * pht)
    Phase = pht
    lC = len(C)
    Param = C
    descending_sequence = np.arange(lC-1, 0, -1)
    IFT_coefficients = np.array(C[:lC-1]) * descending_sequence
    IFT = np.polyval(IFT_coefficients, t)
    return x, Param, IFT, Phase
