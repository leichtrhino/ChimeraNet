
import numpy as np
import sklearn

"""
input: NxTxFxD tensor
output: NxCxFxT tensor
"""
def from_embedding(embedding, n_channels, n_jobs=-1):
    embedding_dim = embedding.shape[-1]
    labels = sklearn.cluster.KMeans(
        n_clusters=n_channels, n_jobs=n_jobs
    ).fit(
        embedding.reshape(embedding.size // embedding_dim, embedding_dim)
    ).labels_
    mask = np.eye(n_channels)[labels]\
        .reshape(list(embedding.shape[:-1])+[n_channels])\
        .transpose((0, 3, 2, 1))
    return mask

"""
input: NxTxFxC tensor
output: NxCxFxT tensor
"""
def from_mask(mask):
    return mask.transpose((0, 3, 2, 1))
