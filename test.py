import numpy as np
from hfmfmgen import hfmfmgen

T=2
N=256

t = np.linspace(-T/2, T/2 - T/N, N)

x, Param, IFT, Phase = hfmfmgen(t)
print(IFT)