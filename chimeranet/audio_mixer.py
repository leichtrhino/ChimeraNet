
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
        self, audio_loader, a_time=(1, 1), a_freq=(0, 0), a_amp=(0, 0)):
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
        
        if self._sync_time_flag and min_len > n_loads * 100:
            # large audio library, make sample_size random indices
            return make_random(n_loads, min_len)\
                .repeat(n_channels).reshape((n_loads, n_channels)).T
        elif self._sync_time_flag and min_len <= n_loads * 100:
            # small audio library, make 0-sample_size array and shuffle
            return make_range_shuffle(n_loads, min_len)\
                .repeat(n_channels).reshape((n_loads, n_channels)).T
        else: # if not self._sync_time_flag
            return [
                make_random(lpc, len(ar))
                if len(ar) > lpc * 100 else
                make_range_shuffle(lpc, len(ar))
                for lpc, ar in zip(loads_per_channel, self._audio_readers)
            ]
        # end function

    def _mod_time(self, audio_list):
        if self._sync_time_flag:
            min_time_rate = max(t[0] for t in self._augment_time_list)
            max_time_rate = min(t[1] for t in self._augment_time_list)
            assert min_time_rate <= max_time_rate,\
                'Invalid time augmentation'
            rates = [random.uniform(min_time_rate, max_time_rate)]\
                * len(audio_list)
            offsets = [random.uniform(0, 1)] * len(audio_list)
        else:
            rates = [
                random.uniform(*t) for t in self._augment_time_list
            ]
            offsets = [random.uniform(0, 1) for _ in self._augment_time_list]
        def mod_single(t):
            a, r, o = t
            n_slice_samples = librosa.core.time_to_samples(
                self._time_in_sec / r, sr=self._sr
            )
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
            return a * 10 ** (r / 10)
        return list(map(mod_single, zip(audio_list, rates)))
    
    def _transform_specs(self, raw_spec_list):
        mod_specs = raw_spec_list
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
        if self._sync_time_flag\
            and any(l != loads_per_channel[0] for l in loads_per_channel):
            raise ValueError('different loads for time sync mode')
        # n_channel x n_loads
        idx = self._make_index_for_reader(loads_per_channel)
        raw_audio_list = [
            [a.load_audio(i, self._sr) for mi, i in enumerate(ii) if mi < loads_per_channel[li]]
            for li, (a, ii) in enumerate(zip(self._audio_readers, idx))
        ]
        mod = lambda a: self._mod_amp(self._mod_freq(self._mod_time(a)))
        raw_audio_list = list(zip(*map(mod, zip(*raw_audio_list))))

        raw_spec_list = [
            [
                np.abs(librosa.core.stft(audio, self._n_fft, self._hop_length))
                for audio in channel
            ]
            for channel in raw_audio_list
        ]
        # n_sample x n_channel
        if self._sync_time_flag:
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
