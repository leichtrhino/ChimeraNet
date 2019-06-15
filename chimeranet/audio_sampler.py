
import random
import librosa
import numpy as np

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
    
    def _make_index_for_reader(self):
        if self._sync_flag:
            idx = [random.randrange(0, min(map(len, self._audio_readers)))]\
                * len(self._audio_readers)
        else:
            idx = [random.randrange(0, len(a)) for a in self._audio_readers]
        return idx
    
    def _stretch_time(self, raw_audio_list):
        if self._sync_flag:
            rates = [
                random.uniform(self._augment_time_lo, self._augment_time_hi)
            ] * len(raw_audio_list)
        else:
            rates = [
                random.uniform(self._augment_time_lo, self._augment_time_hi)
                for _ in range(len(raw_audio_list))
            ]
        return [
            librosa.effects.time_stretch(y, r)
            for y, r in zip(raw_audio_list, rates)
        ]
    
    def _pitch_shift(self, raw_audio_list):
        bins_per_octave = self._n_mels * 8
        if self._sync_flag:
            n_steps = [int(
                random.uniform(self._augment_freq_lo, self._augment_freq_hi)\
                    * bins_per_octave
            )] * len(raw_audio_list)
        else:
            n_steps = [int(
                random.uniform(self._augment_freq_lo, self._augment_freq_hi)\
                    * bins_per_octave
            ) for _ in range(len(raw_audio_list))]
        return [
            librosa.effects.pitch_shift(y, self._sr, n, bins_per_octave)
            for y, n in zip(raw_audio_list, n_steps)
        ]
    
    def _modify_amplitude(self, spectrograms):
        return [
            librosa.core.db_to_amplitude(
                librosa.core.amplitude_to_db(spectrogram)
                + random.uniform(self._augment_amp_lo, self._augment_amp_hi)
            ) for spectrogram in spectrograms
        ]
    
    def _transform_audio(self, raw_audio_list):
        audio_list = self._pitch_shift(self._stretch_time(raw_audio_list))
        specs = [
            np.abs(librosa.core.stft(audio, self._n_fft, self._hop_length))
            for audio in audio_list
        ]
        mod_specs = self._modify_amplitude(specs)
        
        T = librosa.core.time_to_frames(
            self._time_in_sec, sr=self._sr, hop_length=self._hop_length
        )
        n_frames_min = min(x.shape[1] for x in mod_specs)
        if self.sync_flag and n_frames_min > T:
            offset = [random.randrange(0, n_frames_min - T)]\
                * len(raw_audio_list)    
        elif self.sync_flag and n_frames_min <= T:
            offset = [0] * len(raw_audio_list)
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

    def make_speclist(self):
        idx = self._make_index_for_reader()
        raw_audio_list = [
            a.load_audio(i, self._sr) for a, i in zip(self._audio_readers, idx)
        ]
        spec_list = self._transform_audio(raw_audio_list)
        return spec_list

    def generate_speclist(self):
        while True:
            yield self.make_audioset()
