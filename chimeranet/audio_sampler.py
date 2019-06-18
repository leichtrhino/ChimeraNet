
import math
import random
import librosa
import numpy as np
import scipy

from .audio_loader import AudioLoader

class SpectrogramSampler:
    def __init__(self):
        self._audio_readers = []
        self.sync_flag(True)
        self.time(.5)
        self.n_mels(64)
        self.sr(16000)
        self.n_fft(2048)
        self.hop_length(512)
        self.augment_time_range(1, 1)
        self.augment_freq_range(0, 0)
        self.augment_amp_range(0, 0)

    def add_reader(self, audio_reader):
        self._audio_readers.append(audio_reader)
        return self
    
    def add_loader(self, audio_loader):
        return self.add_reader(audio_loader)

    def sync_flag(self, sync_flag):
        self._sync_flag = sync_flag
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

    def augment_time_range(self, lo, hi):
        self._augment_time_lo, self._augment_time_hi = lo, hi
        return self

    def augment_freq_range(self, lo, hi):
        self._augment_freq_lo, self._augment_freq_hi = lo, hi
        return self

    def augment_amp_range(self, lo, hi):
        self._augment_amp_lo, self._augment_amp_hi = lo, hi
        return self
    
    def get_number_of_channels(self):
        return len(self._audio_readers)
    
    def get_frames(self):
        return librosa.core.time_to_frames(
            self._time_in_sec, sr=self._sr, hop_length=self._hop_length
        )
    
    def get_number_of_mel_bins(self):
        return self._n_mels
    
    def _make_index_for_reader(self, loads_per_channel):
        # n_channel x loads_per_channel
        min_len = min(map(len, self._audio_readers))
        n_loads = min(loads_per_channel)
        n_channels = len(self._audio_readers)
        def make_random(N, M):
            return np.random.randint(0, M, size=(N,))
        def make_range_shuffle(N, M):
            idx = np.arange(M).repeat(int(math.ceil(N // M + 1)))
            np.random.shuffle(idx)
            return idx[:N]
        
        if self._sync_flag and min_len > n_loads * 100:
            # large audio library, make sample_size random indices
            return make_random(n_loads, min_len)\
                .repeat(n_channels).reshape((n_loads, n_channels)).T
        elif self._sync_flag and min_len <= n_loads * 100:
            # small audio library, make 0-sample_size array and shuffle
            return make_range_shuffle(n_loads, min_len)\
                .repeat(n_channels).reshape((n_loads, n_channels)).T
        else: # if not self._sync_flag
            return [
                make_random(lpc, len(ar))
                if len(ar) > lpc * 100 else
                make_range_shuffle(lpc, len(ar))
                for lpc, ar in zip(loads_per_channel, self._audio_readers)
            ]
        # end function
    
    def _transform_specs(self, raw_spec_list):
        # transform time, pitch, amplitude
        time_lo, time_hi = self._augment_time_lo, self._augment_time_hi
        if self._sync_flag:
            time_rates = [random.uniform(time_lo, time_hi)] * len(raw_spec_list)
        else:
            time_rates = [
                random.uniform(time_lo, time_hi) for _ in raw_spec_list
            ]
        freq_lo, freq_hi = self._augment_freq_lo, self._augment_freq_hi
        orig_height = min(spec.shape[0] for spec in raw_spec_list)
        if self._sync_flag:
            freq_rates = [
                2 ** random.uniform(freq_lo, freq_hi)
            ] * len(raw_spec_list)
        else:
            freq_rates = [
                2 ** random.uniform(freq_lo, freq_hi) for _ in raw_spec_list
            ]
        mod_specs = [
            scipy.ndimage.zoom(s, (fr, 1/tr), order=0)
            for s, tr, fr in zip(raw_spec_list, time_rates, freq_rates)
        ]
        mod_specs = [
            np.vstack((s, np.zeros((orig_height - s.shape[0], s.shape[1]))))
            if s.shape[0] < orig_height else s[:orig_height, :]
            for s in mod_specs
        ]
        amp_lo, amp_hi = self._augment_amp_lo, self._augment_amp_hi
        mod_specs = [
            s * 10 ** (random.uniform(amp_lo, amp_hi) / 10)
            for s in mod_specs
        ]

        T = librosa.core.time_to_frames(
            self._time_in_sec, sr=self._sr, hop_length=self._hop_length
        )
        n_frames_min = min(x.shape[1] for x in mod_specs)
        if self.sync_flag and n_frames_min > T:
            offset = [random.randrange(0, n_frames_min - T)]\
                * len(raw_spec_list)    
        elif self.sync_flag and n_frames_min <= T:
            offset = [0] * len(raw_spec_list)
        else:
            offset = [
                random.randrange(0, x.shape[1] - T) if x.shape[1] > T else 0
                for x in mod_specs
            ]
        mod_specs = [
            np.hstack((
                s[:, o:min(o+T, s.shape[1])],
                np.zeros((s.shape[0], max(0, o+T-s.shape[1])))
            ))
            for s, o in zip(mod_specs, offset)
        ]
        
        mel_basis = librosa.filters.mel(self._sr, self._n_fft, self._n_mels)
        mel_specs = [np.dot(mel_basis, s**2) for s in mod_specs]
        return mel_specs

    def make_specs(self, sample_size=1, loads_per_channel=1):
        if type(loads_per_channel) == int:
            loads_per_channel = [
                loads_per_channel for _ in range(len(self._audio_readers))
            ]
        if len(loads_per_channel) != len(self._audio_readers):
            raise ValueError('length of loads_per_channel does not match')
        if self._sync_flag\
            and any(l != loads_per_channel[0] for l in loads_per_channel):
            raise ValueError('different loads for time sync mode')
        # n_channel x n_loads
        idx = self._make_index_for_reader(loads_per_channel)
        raw_audio_list = [
            [a.load_audio(i, self._sr) for mi, i in enumerate(ii) if mi < loads_per_channel[li]]
            for li, (a, ii) in enumerate(zip(self._audio_readers, idx))
        ]
        raw_spec_list = [
            [
                np.abs(librosa.core.stft(audio, self._n_fft, self._hop_length))
                for audio in channel
            ]
            for channel in raw_audio_list
        ]
        # n_sample x n_channel
        if self._sync_flag:
            spec_idx = list(range(max(loads_per_channel)))\
                *int(math.ceil(sample_size / max(loads_per_channel)))
            random.shuffle(spec_idx)
            spec_idx = spec_idx[:sample_size]
            spec_idx = [[i]*len(self._audio_readers) for i in spec_idx]
        else:
            spec_idx = [
                [random.randrange(0, lpc) for lpc in loads_per_channel]
                for _ in range(sample_size)
            ]
        return np.stack([
            np.array(self._transform_specs(
                [raw_spec_list[ci][si] for ci, si in enumerate(sidx)]
            ))
            for sidx in spec_idx
        ])

    def generate_specs(self, batch_size=1, loads_per_channel=1, shuffle_batch=1):
        while True:
            rand_idx = np.arange(batch_size * shuffle_batch)
            np.random.shuffle(rand_idx)
            samples = np.concatenate([
                self.make_specs(batch_size, loads_per_channel)
                for _ in range(shuffle_batch)
            ])[rand_idx]
            for i in range(shuffle_batch):
                yield samples[i*batch_size:(i+1)*batch_size, :]
