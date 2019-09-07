import numpy as np

from math import ceil
from itertools import product
from collections import Counter

"""
split into window
"""
def split_window(x, window_length, hop_length=None):
    if hop_length is None:
        hop_length = window_length // 4
    orig_shape = x.shape
    n_windows = ceil((x.shape[-1] - window_length) / hop_length) + 1
    x = x.reshape((x.size // x.shape[-1], x.shape[-1]))
    x = np.hstack((
        x,
        np.zeros((
            x.shape[0], window_length+hop_length*(n_windows-1)-x.shape[-1]
        ))
    ))
    x = np.array([
        x[:, hop_length*i:window_length+hop_length*i]
        for i in range(n_windows)
    ])
    return x.reshape([n_windows]+list(orig_shape[:-1])+[x.shape[-1]])

def merge_windows_mean(x, hop_length=None):
    if hop_length is None:
        hop_length = x.shape[-1] // 4
    n_windows, l_window = x.shape[0], x.shape[-1]
    orig_shape = x.shape
    x = x.reshape((n_windows, x.size//n_windows//l_window, l_window))
    numer = np.zeros((x.shape[1], l_window+hop_length*(n_windows-1)))
    denom = np.zeros((x.shape[1], l_window+hop_length*(n_windows-1)))
    for i in range(n_windows):
        numer[:, i*hop_length:i*hop_length+l_window] += x[i, :, :]
        denom[:, i*hop_length:i*hop_length+l_window] += 1
    x = numer / denom
    return x.reshape(list(orig_shape[1:-1])+[x.shape[-1]])

def merge_windows_most_common(x, hop_length=None):
    if hop_length is None:
        hop_length = x.shape[-1] // 4
    n_windows, l_window = x.shape[0], x.shape[-1]
    orig_shape = x.shape
    x = x.reshape((n_windows, x.size//n_windows//l_window, l_window))
    height = x.size//n_windows//l_window
    width = l_window+hop_length*(n_windows-1)
    show = [[[] for _ in range(width)] for _ in range(height)]
    for i in range(n_windows):
        for ri, ci in product(range(height), range(l_window)):
            show[ri][i*hop_length+ci].append(x[i, ri, ci])
    x = np.array([
        [
            Counter(show[ri][ci]).most_common(1)[0][0]
            for ci in range(width)
        ]
        for ri in range(height)
    ])
    return x.reshape(list(orig_shape[1:-1])+[x.shape[-1]])
