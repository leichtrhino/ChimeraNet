
import numpy as np
from numpy.linalg import pinv
from librosa.core import stft, istft, magphase
from librosa.filters import mel

def to_feature(y, mel_basis=None, return_phase=False, **kwargs):
    sr = kwargs.get('sr', 16000)
    n_fft = kwargs.get('n_fft', 512)
    hop_length = kwargs.get('hop_length', 128)
    n_mels = kwargs.get('n_mels', 128)
    if mel_basis is None:
        mel_basis = mel(sr, n_fft, n_mels)
    S, phase = magphase(stft(y, n_fft, hop_length))
    Sf = np.dot(mel_basis, S)
    if return_phase:
        return Sf, phase
    else:
        return Sf

def from_feature(Sf, phase, mel_basis_inv=None, **kwargs):
    sr = kwargs.get('sr', 16000)
    n_fft = kwargs.get('n_fft', 512)
    #hop_length = kwargs.get('hop_length', 128)
    n_mels = kwargs.get('n_mels', 128)
    if mel_basis_inv is None:
        mel_basis_inv = pinv(mel(sr, n_fft, n_mels))
    return istft(np.dot(mel_basis_inv, Sf) * phase)
