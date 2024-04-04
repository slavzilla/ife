import numpy as np
from hfmfmgen import hfmfmgen

lol = range(1, 4097)


res = np.fft.fftshift(np.fft.fft(lol, 512))
print(res)