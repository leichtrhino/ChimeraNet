from math import ceil, floor, log
from itertools import product
import numpy as np
import scipy.linalg
import scipy.signal
import sklearn.mixture

def gen_psi(K, M, N):
    frame_size = 2 * (K-1)
    hop_length = int((N - frame_size) / M)
    DFT = scipy.linalg.dft((K-1)*2)[:K, :]
    hann = scipy.signal.hann((K-1)*2, sym=False)
    psi = lambda k, m: np.hstack((
        np.zeros(m*hop_length),
        hann * DFT[k],
        np.zeros(max(0, N-m*hop_length-frame_size))
    ))[:N]
    return psi

def mel_mat(R, K, sr, fmin=0., fmax=None):
    if fmax is None:
        fmax = sr / 2
    n_fft = 2 * (K-1)
    m = lambda f: 1125 * log(1 + f / 700)
    mi = lambda m: 700 * (np.e ** (m / 1125) - 1)
    fs = mi(np.linspace(m(fmin), m(fmax), R+2))
    fi = np.floor((n_fft+1) * fs / sr).astype(int)[:, None]
    k = np.arange(K)
    F = np.maximum(np.where(
        k <= fi[1:-1],
        (k - fi[:-2]) / (fi[1:-1] - fi[:-2])**2,
        (fi[2:] - k) / (fi[2:] - fi[1:-1])**2
    ), 0)
    return F

def idct_mat(Q, R):
    return np.hstack((
        np.full((Q, 1), 1/2),
        np.cos(np.outer(np.arange(Q)+1/2, np.arange(1, R)/R*np.pi))
    ))

class MaskRestorator:
    def __init__(mixture):
        assert type(mixture) is sklearn.mixture.GaussianMixture
        self.psi = gen_psi(self.K, self.M, self.N)
        self.mel_mat = mel_mat(self.R, self.K)
        self.idct_mat = idct_mat(self.Q, self.R)
    pass

