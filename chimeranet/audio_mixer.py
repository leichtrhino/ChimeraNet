
import sys
import os
import multiprocessing
import math
import random
import librosa
import numpy as np
import scipy

from .audio_loader import AudioLoader

class AudioMixer:
    def __init__(self):
        self._audio_readers = []
        self._augment_time_list = []
        self._augment_freq_list = []
        self._augment_amp_list = []
        self.time(.5)
        self.n_mels(64)
        self.sr(16000)
        self.n_fft(2048)
        self.hop_length(512)
        self.sync_flag(True)
    
    def add_loader(
        self, audio_loader, a_time=(0, 0), a_freq=(0, 0), a_amp=(0, 0)):
        self._audio_readers.append(audio_loader)
        self._augment_time_list.append(a_time)
        self._augment_freq_list.append(a_freq)
        self._augment_amp_list.append(a_amp)
        return self

    def sync_flag(self, flag):
        self.sync_time_flag(flag)
        self.sync_freq_flag(flag)
        self.sync_amp_flag(flag)
        return self
    
    def sync_time_flag(self, flag):
        self._sync_time_flag = flag
        return self

    def sync_freq_flag(self, flag):
        self._sync_freq_flag = flag
        return self

    def sync_amp_flag(self, flag):
        self._sync_amp_flag = flag
        return self

    def time(self, t):
        self._time_in_sec = t
        return self
    
    def n_mels(self, f):
        self._n_mels = f
        return self
    
    def sr(self, s):
        self._sr = s
        return self
    
    def n_fft(self, n):
        self._n_fft = n
        return self
    
    def hop_length(self, h):
        self._hop_length = h
        return self

    def get_number_of_channels(self):
        return len(self._audio_readers)
    
    def get_frames(self):
        return librosa.core.time_to_frames(
            self._time_in_sec, sr=self._sr, hop_length=self._hop_length
        )
    
    def get_number_of_mel_bins(self):
        return self._n_mels
    
    def _make_index_for_reader(self, n_samples):
        # n_samples x n_channels
        min_len = min(map(len, self._audio_readers))
        n_channels = len(self._audio_readers)
        def make_range_shuffle(N, M):
            idx_sub = np.arange(M)
            np.random.shuffle(idx_sub)
            idx = np.hstack((
                np.arange(M).repeat(int(math.ceil(N // M))).flatten(),
                idx_sub[:N%M]
            ))
            np.random.shuffle(idx)
            return idx
        
        if self._sync_time_flag:
            # small audio library, make 0-sample_size array and shuffle
            return make_range_shuffle(n_samples, min_len)\
                .repeat(n_channels).reshape((n_samples, n_channels))
        else: # if not self._sync_time_flag
            return np.array([
                make_range_shuffle(n_samples, len(ar))
                for ar in self._audio_readers
            ]).T
        # end function

    def _mod_time(self, audio_list):
        if self._sync_time_flag:
            min_time_rate = max(t[0] for t in self._augment_time_list)
            max_time_rate = min(t[1] for t in self._augment_time_list)
            assert min_time_rate <= max_time_rate,\
                'Invalid time augmentation'
            rates = [2**random.uniform(min_time_rate, max_time_rate)]\
                * len(audio_list)
            offsets = [random.uniform(0, 1)] * len(audio_list)
        else:
            rates = [
                2**random.uniform(*t) for t in self._augment_time_list
            ]
            offsets = [random.uniform(0, 1) for _ in self._augment_time_list]
        def mod_single(t):
            a, r, o = t
            n_slice_samples = librosa.core.time_to_samples(
                self._time_in_sec * r, sr=self._sr
            ) + self._hop_length
            offset = max(0, int(o * (a.size - n_slice_samples)))
            # slice
            sliced = a if a.size < offset + n_slice_samples\
                else a[offset:offset+n_slice_samples]
            # stretch
            stretched = librosa.effects.time_stretch(sliced, r)
            return stretched
        return list(map(mod_single, zip(audio_list, rates, offsets)))

    def _mod_freq(self, audio_list):
        if self._sync_freq_flag:
            min_freq_rate = max(t[0] for t in self._augment_freq_list)
            max_freq_rate = min(t[1] for t in self._augment_freq_list)
            assert min_freq_rate <= max_freq_rate,\
                'Invalid freq augmentation'
            rates = [random.uniform(min_freq_rate, max_freq_rate)]\
                * len(audio_list)
        else:
            rates = [random.uniform(*r) for r in self._augment_freq_list]
        bins_per_octave = self._n_mels * 8
        def mod_single(t):
            a, r = t
            n_step = int(r * bins_per_octave)
            return librosa.effects.pitch_shift(
                a, self._sr, n_step, bins_per_octave
            )
        return list(map(mod_single, zip(audio_list, rates)))
    
    def _mod_amp(self, audio_list):
        if self._sync_amp_flag:
            min_amp_rate = max(t[0] for t in self._augment_amp_list)
            max_amp_rate = min(t[1] for t in self._augment_amp_list)
            assert min_amp_rate <= max_amp_rate,\
                'Invalid freq augmentation'
            rates = [random.uniform(min_amp_rate, max_amp_rate)]\
                * len(audio_list)
        else:
            rates = [random.uniform(*r) for r in self._augment_amp_list]
        def mod_single(t):
            a, r = t
            S = librosa.core.stft(a, self._n_fft, self._hop_length)
            Sa, Sp = librosa.core.magphase(S)
            Sa = (Sa / max(np.max(Sa), 1e-32))**2 * 10**(r/10)
            a = librosa.core.istft(Sa**0.5 * Sp, self._hop_length)
            return np.clip(a, -1., 1.)
        return list(map(mod_single, zip(audio_list, rates)))
    
    def _transform_specs(self, audio_list):
        T = librosa.core.time_to_frames(
            self._time_in_sec, sr=self._sr, hop_length=self._hop_length
        )
        raw_spec_list = [
            np.abs(librosa.core.stft(audio, self._n_fft, self._hop_length))
            for audio in audio_list
        ]
        mod_specs = [
            np.hstack((
                s[:, :min(T, s.shape[1])],
                np.zeros((s.shape[0], max(0, T-s.shape[1])))
            ))
            for s in raw_spec_list
        ]
        
        mel_basis = librosa.filters.mel(self._sr, self._n_fft, self._n_mels)
        mel_specs = [np.dot(mel_basis, s**2) for s in mod_specs]
        mel_specs = [np.clip(s - np.min(s), 0, 1) for s in mel_specs]
        return mel_specs
    
    def make_single_specs(self, idx):
        raw_audio_list = [
            a.load_audio(i, self._sr) for a, i in zip(self._audio_readers, idx)
        ]
        mod = lambda a: self._mod_amp(self._mod_freq(self._mod_time(a)))
        return self._transform_specs(mod(raw_audio_list))

    def make_specs(self, sample_size=1, n_jobs=-1):
        # n_samples x n_channels
        if n_jobs <= 0:
            n_jobs = os.cpu_count()
        idx = self._make_index_for_reader(sample_size)
        if n_jobs > 1:
            p = multiprocessing.Pool(n_jobs, maxtasksperchild=1)
            samples = np.array(list(p.map(self.make_single_specs, idx)))
            p.close()
            p.join()
        else:
            samples = np.array(list(map(self.make_single_specs, idx)))
        return samples

    def generate_specs(self, batch_size=1, n_batch=1, n_jobs=-1):
        while True:
            rand_idx = np.arange(batch_size * shuffle_batch)
            np.random.shuffle(rand_idx)
            samples = self.make_specs(batch_size * n_batch, n_jobs)[rand_idx]
            for i in range(n_batch):
                yield samples[i*batch_size:(i+1)*batch_size, :]
