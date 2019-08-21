
import numpy as np
from scipy.special import softmax

"""
input: NxCxFxT tensor
output: NxTxF tensor
"""
def to_mixture(multichannels):
    C = multichannels.shape[1]
    return multichannels.sum(axis=1).transpose((0, 2, 1))

"""
input: NxCxFxT tensor
output: 'embedding': NxTxFxC tensor
        'mask': NxTxFx(C+1) tensor
"""
def to_true_pair(multichannels, mask='linear', threshold=None):
    mixture = np.expand_dims(to_mixture(multichannels), axis=-1)
    mask_func = _binary_mask if mask == 'binary' else\
        _softmax_mask if mask == 'softmax' else _linear_mask
    return {
        'embedding': _binary_mask(multichannels, threshold).transpose((0, 3, 2, 1)),
        'mask': np.concatenate(
            (mask_func(multichannels, threshold).transpose((0, 3, 2, 1)), mixture),
            axis=-1
        )
    }

"""
input: NxCxFxT tensor
output: NxCxFxT tensor
"""
def _binary_mask(multichannels, threshold=None):
    m = multichannels.transpose((0, 3, 2, 1)) # => NxTxFxC
    C = m.shape[-1]
    masks = np.eye(C)[
        m.reshape((m.size // C, C)).argmax(axis=-1)
    ].reshape(m.shape).transpose((0, 3, 2, 1)) # => NxCxFxT
    if threshold is not None and threshold > 0:
        masks = np.where(multichannels > threshold, masks, 0)
    return masks

"""
input: NxCxFxT tensor
output: NxCxFxT tensor
"""
def _linear_mask(multichannels, threshold=None):
    m = multichannels
    C = m.shape[1]
    mixture = np.expand_dims(m.sum(axis=1), 1)
    masks = np.clip(m / np.maximum(mixture, 1e-32), 0, 1)
    if threshold is not None and threshold > 0:
        masks = np.where(multichannels > threshold, masks, 0)
    return masks

"""
input: NxCxFxT tensor
output: NxCxFxT tensor
"""
def _softmax_mask(multichannels, threshold=None):
    masks = softmax(multichannels, axis=1)
    if threshold is not None and threshold > 0:
        masks = np.where(multichannels > threshold, masks, 0)
    return masks
