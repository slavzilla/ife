import numpy as np

def hfmfmgen(t):
    dt = t[1]-t[0]
    wmax = 2 * np.pi * 0.5 / dt
    cm = 64 * np.pi
    c = []
    r = 1
    C = []
    mif = 0
    while r <= 5:
        c = [cm * (np.random.rand() - 1/2)] + c
        pht = np.polyval(c, t)
        if np.min(pht) < 0:
            c[-1] = c[-1] - np.min(pht) + 0.1
        r = r + 1
        C = c.copy()
    pht = np.real(np.polyval(C, t))
    if np.min(pht) < 0:
        pht = pht - np.min(pht + 1)
        C[-1] = C[-1] + 1
    pht = np.log(pht)
    ift = np.diff(pht) / dt
    d = np.max(ift) - np.min(ift)
    scal = (np.random.rand() * 0.9 + 0.05) * 2 * wmax / d
    phase = scal * np.log(np.real(np.polyval(C, t)))
    a1 = 0
    ift = np.diff(phase) / dt
    if np.min(ift) < -wmax:
        phase = phase + (-wmax - np.min(ift)) * t
        a1 = (-wmax - np.min(ift))
    elif np.max(ift) > wmax:
        phase = phase - (np.max(ift) - wmax) * t
        a1 = -(np.max(ift) - wmax)
    x = np.exp(1j * phase)
    Phase = phase
    rC = len(C)
    descending_sequence = np.arange(rC-1, 0, -1)
    derivative_coeffs = np.array(C[:-1]) * descending_sequence
    ift_derivative = scal * np.real(np.polyval(derivative_coeffs, t))
    original_poly = np.real(np.polyval(C, t))
    IFT = ift_derivative / original_poly + a1
    Param = [scal, C, a1]
    return x, Param, IFT, Phase