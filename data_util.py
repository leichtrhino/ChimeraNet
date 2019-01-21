
import numpy as np
from math import ceil
from itertools import product
from collections import Counter
from sklearn.cluster import KMeans

def split_window(X, window_length, hop_length=None):
    if hop_length is None:
        hop_length = window_length // 4
    if len(X.shape) == 3:
        X = X.reshape(X.shape[1:])
    n_windows = ceil((X.shape[1] - window_length) / hop_length) + 1
    Y = []
    for i in range(n_windows):
        valid_length = window_length if window_length + hop_length*i < X.shape[1]\
                       else window_length + hop_length*i - X.shape[1]
        Y.append(np.hstack((
            X[:, hop_length*i:hop_length*i+valid_length],
            np.zeros((X.shape[0], window_length - valid_length))
        )))
    return np.array(Y)

def merge_window_mean(Y, hop_length=None):
    if hop_length is None:
        hop_length = Y.shape[2] // 4
    X = np.zeros((Y.shape[1], Y.shape[2]+hop_length*(Y.shape[0]-1)))
    denom = np.zeros(X.shape)
    for i in range(Y.shape[0]):
        denom[:, i*hop_length:i*hop_length+Y.shape[2]] += 1
        X[:, i*hop_length:i*hop_length+Y.shape[2]] += Y[i, :, :]
    return X / denom

def merge_window_most_common(Y, hop_length=None):
    if hop_length is None:
        hop_length = Y.shape[2] // 4
    row, col= Y.shape[1], Y.shape[2]+hop_length*(Y.shape[0]-1)
    X = [[[] for _ in range(col)] for _ in range(row)]
    for i in range(Y.shape[0]):
        for ri, ci in product(range(row), range(Y.shape[2])):
            X[ri][i*hop_length+ci].append(Y[i, ri, ci])
    X = np.array([
        [Counter(X[ri][ci]).most_common(1)[0][0] for ci in range(col)]
        for ri in range(row)])
    return X

"""
data conversion functions
"""
def to_embd_matrix(*args):
    Ys = []
    for Xs in zip(*args):
        Xs = np.stack(Xs)
        am = np.argmax(Xs, axis=0)
        Y = np.zeros(Xs.shape)
        for i in range(Xs.shape[0]):
            Y[i, am==i] = 1
        Ys.append(Y)
    return np.array(Ys)

def to_msa_mask(*args):
    Ys = []
    for Xs in zip(*args):
        Xs = np.stack(X for X in Xs)
        Ys.append(Xs)
    return np.array(Ys)

def to_mmsa_mask(*args):
    Ys = []
    for Xs in zip(*args):
        Os, S = Xs[:-1], Xs[-1]
        Os = np.stack(Os)
        am = np.argmax(Os, axis=0)
        Y = np.zeros(Os.shape)
        for i in range(Os.shape[0]):
            Y[i, am==i] = 1
        Ys.append(np.concatenate((Y, S[np.newaxis, :])))
    return np.array(Ys)

def find_embd_matrix(V, C=2, hop_length=None):
    # produces CxFxTime array
    if hop_length is None:
        hop_length = V.shape[1] // 4
    ns, T, F, D = V.shape
    labels = KMeans(n_clusters=C).fit(
        V.transpose((0, 2, 1, 3)).reshape(ns*F*T, D)
    ).labels_.reshape((ns, F, T))
    V = merge_window_most_common(labels, )
    embd_matrix = []
    for ci in range(C):
        Dc = np.zeros(V.shape)
        Dc[V==ci] = 1
        embd_matrix.append(Dc)
    return np.array(embd_matrix)

def find_mask(M, hop_length=None):
    # produces CxFxTime array
    if hop_length is None:
        hop_length = M.shape[1] // 4
    mask = []
    for M_ci in M.transpose((3, 0, 2, 1)):
        mask.append(merge_window_mean(M_ci, hop_length))
    return np.array(mask)
