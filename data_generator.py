
import os
import numpy as np
import librosa

from random import choice, gauss, uniform
from itertools import product

from feature import to_feature

def dc_true(*args):
    Ys = []
    for Xs in zip(*args):
        Xs = np.stack(Xs)
        am = np.argmax(Xs, axis=0)
        Y = np.zeros(Xs.shape)
        for i in range(Xs.shape[0]):
            Y[i, am==i] = 1
        Ys.append(Y)
    return np.array(Ys)

def msa_true(*args):
    Ys = []
    for Xs in zip(*args):
        Xs = np.stack(X for X in Xs)
        Ys.append(Xs)
    return np.array(Ys)

def mmsa_true(*args):
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

def generate_test_data(voc_dir, mel_dir, k=1, **kwargs):
    sr = kwargs.get('sr', 16000)
    n_fft = kwargs.get('n_fft', 512)
    hop_length = kwargs.get('hop_length', 128)
    n_mels = kwargs.get('n_mels', 128)
    duration = kwargs.get('duration', None)
    params = {'n_fft': n_fft, 'hop_length': hop_length, 'n_mels': n_mels}

    mel_basis = librosa.filters.mel(sr, n_fft, n_mels)

    samples = librosa.core.time_to_samples(duration, sr)
    filter_pred = lambda f: librosa.core.get_duration(filename=f) >= duration
    get_path = lambda dname: map(lambda f: os.path.join(dname, f), os.listdir(dname))
    voc_files = list(filter(filter_pred, get_path(voc_dir)))
    mel_files = list(filter(filter_pred, get_path(mel_dir)))

    def generate_one():
        mix = None
        while mix is None:
            voc_file = choice(voc_files)
            voc_offset = uniform(0, librosa.core.get_duration(filename=voc_file)-duration)
            voc, _ = librosa.core.load(voc_file, sr=sr, offset=voc_offset, duration=duration)

            mel_file = choice(mel_files)
            mel_offset = uniform(0, librosa.core.get_duration(filename=mel_file)-duration)
            mel, _ = librosa.core.load(mel_file, sr=sr, offset=mel_offset, duration=duration)
            if voc.size != samples or mel.size != samples:
                continue

            random_loudness = lambda x: np.e**(gauss(0, 1)) * x
            clip = lambda x: np.clip(x, -1, 1)
            voc = clip(random_loudness(voc))
            mel = clip(random_loudness(mel))
            mix = clip(voc + mel)

        Voc, VocP = to_feature(voc, mel_basis, return_phase=True, **params)
        Mel, MelP = to_feature(mel, mel_basis, return_phase=True, **params)
        Mix, MixP = to_feature(mix, mel_basis, return_phase=True, **params)
        return (Voc, VocP), (Mel, MelP), (Mix, MixP)

    while True:
        raws = np.transpose(np.array([generate_one() for _ in range(k)]), (1, 2, 0))
        ((Voc, VocP), (Mel, MelP), (Mix, MixP)) = raws
        Voc, VocP = np.stack(Voc), np.stack(VocP)
        Mel, MelP = np.stack(Mel), np.stack(MelP)
        Mix, MixP = np.stack(Mix), np.stack(MixP)
        raws = ((Voc, VocP), (Mel, MelP), (Mix, MixP))
        dc = dc_true(Voc, Mel)
        mi = mmsa_true(Voc, Mel, Mix)
        yield raws, {'V': dc, 'M': mi}

def generate_data(voc_dir, mel_dir, k=1, **kwargs):
    for raws, targets in generate_test_data(voc_dir, mel_dir, k, **kwargs):
        yield raws[2][0], targets        
