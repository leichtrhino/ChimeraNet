#!/usr/env/bin python
import os, sys
import numpy as np
import librosa

from itertools import permutations

from keras.models import load_model
from keras.utils import CustomObjectScope

sys.path.append(os.path.join(os.path.split(__file__)[0], '..'))
from chimeranet.model import ChimeraNetModel
from chimeranet.postprocessing import from_embedding, from_mask
from chimeranet.window_util import split_window, merge_windows_mean, merge_windows_most_common

def main():
    # parameters and chimeranet model
    time = 0.5
    sr, n_fft, hop_length, n_mels = 16000, 512, 128, 64
    model_path = 'model_trisep.h5'
    audio_path = 'ep01.mono.wav'
    output_prefix = 'out'
    embd_name = 'embd'
    mask_name = 'mask'
    n_channels = 3 # if trisep task, 2 if dsd100 task
    d_embedding = 20
    
    T, F = librosa.core.time_to_frames(time, sr, hop_length), n_mels
    cm = ChimeraNetModel(T, F, n_channels, d_embedding)

    # load audio and split into windows
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels)
    audio, _ = librosa.core.load(audio_path, sr=sr)
    audio = audio[:100000]
    spec, phase = librosa.core.magphase(
        librosa.core.stft(audio, n_fft, hop_length)
    )
    mel_spec = np.dot(mel_basis, spec)
    x = split_window(mel_spec, T).transpose((0, 2, 1))

    # load actual model and predict
    with CustomObjectScope({
        '_loss_deepclustering': cm.loss_deepclustering(),
        '_loss_mask': cm.loss_mask(),
    }):
        model = load_model(model_path)
    embedding, mask = model.predict(x)
    y_embd = from_embedding(embedding, n_channels)
    y_mask = from_mask(mask)
    mask_embd = merge_windows_most_common(y_embd)[:, :, :mel_spec.shape[1]]
    mask_mask = merge_windows_mean(y_mask)[:, :, :mel_spec.shape[1]]

    # reconstruct from prediction
    mel_basis_inv = np.linalg.pinv(mel_basis)
    for ci, mask in enumerate(mask_mask):
        pred_spec = np.dot(mel_basis_inv, mask*mel_spec)
        out_audio = librosa.core.istft(pred_spec*phase, hop_length)
        librosa.output.write_wav(
            '{}-{}-{}.wav'.format(output_prefix, mask_name, ci),
            out_audio, sr
        )
    # to align embedding to mask
    mask_embd_ordered = min(
        (t for t in permutations(mask_embd)),
        key=lambda t: np.sum(np.abs(np.array(t)-mask_mask))
    )
    for ci, mask in enumerate(mask_embd_ordered):
        pred_spec = np.dot(mel_basis_inv, mask*mel_spec)
        out_audio = librosa.core.istft(pred_spec*phase, hop_length)
        librosa.output.write_wav(
            '{}-{}-{}.wav'.format(output_prefix, embd_name, ci),
            out_audio, sr
        )

if __name__ == '__main__':
    main()
