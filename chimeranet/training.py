
import os
import multiprocessing
import numpy as np
import librosa
from scipy.special import softmax

def _to_training_data_single(args):
    cs, T, F, sr, n_fft, hop_length, noise, threshold = args

    mel_basis = librosa.filters.mel(sr, n_fft, F, norm=None)
    def to_feature(y):
        return np.dot(
            mel_basis,
            np.abs(librosa.stft(y, n_fft, hop_length))
        )
    C = len(cs)
    m = sum(c*np.max(c) for c in cs)\
        / max(sum(np.max(c) for c in cs), 1e-16)
    mixture = to_feature(m)[:, :T]
    channels = np.array([to_feature(c) for c in cs])[:, :, :T]
    if noise:
        channels += np.random.uniform(0, noise, size=channels.shape)

    c = channels.transpose((2, 1, 0)) # => TxFxC
    binary_mask = np.eye(C)[
        c.reshape((c.size // C, C)).argmax(axis=-1)
    ].reshape(c.shape).transpose((2, 1, 0)) # => CxFxT
    if threshold is not None and threshold > 0:
        binary_mask = np.where(channels > threshold, binary_mask, 0)

    softmax_mask = softmax(channels, axis=0)
    if threshold is not None and threshold > 0:
        softmax_mask = np.where(channels > threshold, softmax_mask, 0)

    x = mixture.T
    y_embedding = binary_mask.transpose((2, 1, 0))
    y_mask = np.dstack((
        softmax_mask.transpose((2, 1, 0)),
        mixture.T[:, :, None]
    ))
    return x, dict(embedding=y_embedding, mask=y_mask)

def to_training_data_single(
    cs, T, F, sr=44100, n_fft=2048, hop_length=None,
    noise=None, threshold=None
):
    return _to_training_data_single(
        (cs, T, F, sr, n_fft, hop_length, noise, threshold)
    )

"""
input: NxCxFxT tensor
output: 'embedding': NxTxFxC tensor
        'mask': NxTxFx(C+1) tensor
"""
def to_training_data(
    ys, T, F, sr=44100, n_fft=2048, hop_length=None,
    noise=None, threshold=None,
    n_jobs=1
):
    args_list = [
        (cs, T, F, sr, n_fft, hop_length, noise, threshold)
        for cs in ys
    ]
    if n_jobs <= 0:
        n_jobs = os.cpu_count()
    if n_jobs > 1:
        p = multiprocessing.Pool(n_jobs, maxtasksperchild=1)
        data = list(p.map(_to_training_data_single, args_list))
        p.close()
        p.join()
    else:
        data = list(map(_to_training_data_single, args_list))
    return np.array([x for x, _ in data]),\
        dict(
            (k, np.array([y[k] for _, y in data]))
            for k in ('embedding', 'mask')
        )
