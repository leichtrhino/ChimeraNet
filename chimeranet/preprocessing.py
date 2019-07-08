
import numpy as np

"""
input: NxCxFxT tensor
output: NxTxF tensor
"""
def to_mixture(multichannels):
    C = multichannels.shape[1]
    return multichannels.sum(axis=1).transpose((0, 2, 1)) / C

"""
input: NxCxFxT tensor
output: 'embedding': NxTxFxC tensor
        'mask': NxTxFx(C+1) tensor
"""
def to_true_pair(multichannels):
    m = multichannels.transpose((0, 3, 2, 1)) # => NxTxFxC
    C = m.shape[-1]
    mixture = np.expand_dims(m.sum(axis=-1) / C, -1) # => NxTxFx1
    e_masks = np.eye(C)[
        m.reshape((m.size // C, C)).argmax(axis=-1)
    ].reshape(m.shape) # => NxTxFxC
    m_masks = np.clip(m / mixture / C, 0, 1)
    return {
        'embedding': e_masks,
        'mask': np.concatenate((m_masks, mixture), axis=-1)
    }

"""
input: NxCxFxT tensor
output: NxTxFxC tensor
"""
def to_true_deepclustering(multichannels):
    m = multichannels.transpose((0, 3, 2, 1)) # => NxTxFxC
    C = m.shape[-1]
    masks = np.eye(C)[
        m.reshape((m.size // C, C)).argmax(axis=-1)
    ].reshape(m.shape) # => NxTxFxC
    return masks

"""
input: NxCxFxT tensor
output: NxTxFx(C+1) tensor
"""
def to_true_mask(multichannels):
    m = multichannels.transpose((0, 3, 2, 1)) # => NxTxFxC
    C = m.shape[-1]
    mixture = np.expand_dims(m.sum(axis=-1) / C, -1) # => NxTxFx1
    masks = np.clip(m / mixture / C, 0, 1)
    return np.concatenate((masks, mixture), axis=-1)
