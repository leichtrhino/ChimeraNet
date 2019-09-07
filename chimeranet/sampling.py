
import os
import math
import random
import multiprocessing
import numpy as np
import librosa

from .datasets import Dataset

def normalise_amplitude(y, amplitude_factor):
    return 2**(amplitude_factor-1) * y / max(np.max(np.abs(y)), 1e-16)

def pitch_shift(y, shift_factor, sr):
    bins_per_octave = 12
    n_steps = int(shift_factor * bins_per_octave)
    return librosa.effects.pitch_shift(y, sr, n_steps)

def time_stretch(y, stretch_factor):
    return librosa.effects.time_stretch(y, 2**stretch_factor)

def transform(y, sr, **kwargs):
    amplitude_factor = kwargs.get('amplitude_factor', 0)
    shift_factor = kwargs.get('shift_factor', 0)
    stretch_factor = kwargs.get('stretch_factor', 0)
    na = lambda y: normalise_amplitude(y, amplitude_factor)
    ts = lambda y: time_stretch(y, stretch_factor)
    ps = lambda y: pitch_shift(y, shift_factor, sr)
    return ts(ps(na(y)))

class Sampler:
    def __init__(self):
        self.duration = 2.
        self.samplerate = 44100
        self.amplitude_factor = (0, 0)
        self.stretch_factor = (0, 0)
        self.shift_factor = (0, 0)

    @property
    def duration(self):
        return self._duration
    @duration.setter
    def duration(self, value):
        self._duration = float(value)

    @property
    def samplerate(self):
        return self._samplerate
    @samplerate.setter
    def samplerate(self, value):
        self._samplerate = int(value)

    @property
    def amplitude_factor(self):
        return self._amplitude_factor
    @amplitude_factor.setter
    def amplitude_factor(self, value):
        try:
            value = float(value)
            value = (value, value)
        except TypeError:
            if not hasattr(value, '__len__') or not 0 < len(value) <= 2:
                raise TypeError
            if len(value) == 1:
                value = next(iter(value))
                value = (value, value)
        self._amplitude_factor = value

    @property
    def stretch_factor(self):
        return self._stretch_factor
    @stretch_factor.setter
    def stretch_factor(self, value):
        try:
            value = float(value)
            value = (value, value)
        except TypeError:
            if not hasattr(value, '__len__') or not 0 < len(value) <= 2:
                raise TypeError
            if len(value) == 1:
                value = next(iter(value))
                value = (value, value)
        self._stretch_factor = value

    @property
    def shift_factor(self):
        return self._shift_factor
    @shift_factor.setter
    def shift_factor(self, value):
        try:
            value = float(value)
            value = (value, value)
        except TypeError:
            if not hasattr(value, '__len__') or not 0 < len(value) <= 2:
                raise TypeError
            if len(value) == 1:
                value = next(iter(value))
                value = (value, value)
        self._shift_factor = value

    def dataset_size(self):
        return 0
    
    def sample(self, n_samples=1, n_jobs=1):
        return [None] * n_samples

    def generate(self, batch_size=1, n_batch_per_round=1, n_jobs=1):
        while True:
            rand_idx = np.arange(batch_size * n_batch_per_round)
            np.random.shuffle(rand_idx)
            samples = self.sample(
                batch_size * n_batch_per_round, n_jobs)[rand_idx]
            for i in range(n_batch):
                yield samples[i*batch_size:(i+1)*batch_size]

class DatasetSampler(Sampler):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def dataset_size(self):
        return len(self.dataset)
    
    def load(self, index, **kwargs):
        offset = kwargs.get('offset', random.uniform(0, 1))
        amplitude_factor = kwargs.get(
            'amplitude_factor', random.uniform(*self.amplitude_factor))
        shift_factor = kwargs.get(
            'shift_factor', random.uniform(*self.shift_factor))
        stretch_factor = kwargs.get(
            'stretch_factor', random.uniform(*self.stretch_factor))

        load_duration = self.duration * (2**stretch_factor)
        y = self.dataset.load(
            index, self.samplerate, offset=offset, duration=load_duration)
        y = transform(
            y, self.samplerate,
            amplitude_factor=amplitude_factor,
            shift_factor=shift_factor,
            stretch_factor=stretch_factor
        )
        samples = librosa.time_to_samples(self.duration, sr=self.samplerate)
        if y.size < samples:
            y = np.hstack((y, np.zeros((samples - y.size,))))
        elif y.size > samples:
            y = y[:samples]
        return y[None, :]

    def sample(self, n_samples=1, n_jobs=1):
        m = self.dataset_size()

        # generate random index
        idx = np.arange(m)
        np.random.shuffle(idx)
        idx = np.hstack((
            np.arange(m).repeat(int(math.ceil(n_samples // m))).flatten(),
            idx[:n_samples%m]
        ))
        np.random.shuffle(idx)

        # load
        if n_jobs <= 0:
            n_jobs = os.cpu_count()
        if n_jobs > 1:
            p = multiprocessing.Pool(n_jobs, maxtasksperchild=1)
            samples = np.array(list(p.map(self.load, idx)))
            p.close()
            p.join()
        else:
            samples = np.array(list(map(self.load, idx)))
        return samples

class AggregateSampler(Sampler):
    def __init__(self, *samplers):
        self.samplers = []
        for s in samplers:
            if isinstance(s, Dataset):
                s = DatasetSampler(s)
            if not isinstance(s, Sampler):
                raise TypeError
            self.samplers.append(s)
        super().__init__()

    @property
    def duration(self):
        durations = tuple(s.duration for s in self.samplers)
        duration = durations[0]
        if any(duration != d for d in durations):
            raise ValueError
        return duration
    @duration.setter
    def duration(self, value):
        for s in self.samplers:
            s.duration = value

    @property
    def samplerate(self):
        samplerates = tuple(s.samplerate for s in self.samplers)
        samplerate = samplerates[0]
        if any(samplerate != s for s in samplerates):
            raise ValueError
        return samplerate
    @samplerate.setter
    def samplerate(self, value):
        for s in self.samplers:
            s.samplerate = value
    
    @property
    def amplitude_factor(self):
        return None
    @amplitude_factor.setter
    def amplitude_factor(self, value):
        pass

    @property
    def stretch_factor(self):
        return None
    @stretch_factor.setter
    def stretch_factor(self, value):
        pass

    @property
    def shift_factor(self):
        return None
    @shift_factor.setter
    def shift_factor(self, value):
        pass
    
    def __getitem__(self, key):
        return self.samplers[key]

class AsyncSampler(AggregateSampler):
    def dataset_size(self):
        return max(s.dataset_size() for s in self.samplers)
    def sample(self, n_samples=1, n_jobs=1):
        return np.hstack([s.sample(n_samples, n_jobs) for s in self.samplers])

class SyncSampler(AggregateSampler):
    def load(self, index, **kwargs):
        offset = kwargs.get('offset', random.uniform(0, 1))
        amplitude_factor = kwargs.get(
            'amplitude_factor', random.uniform(*self.amplitude_factor))
        shift_factor = kwargs.get(
            'shift_factor', random.uniform(*self.shift_factor))
        stretch_factor = kwargs.get(
            'stretch_factor', random.uniform(*self.stretch_factor))

        ys = np.array([
            s.load(
                index, offset=offset,
                amplitude_factor=amplitude_factor,
                shift_factor=shift_factor,
                stretch_factor=stretch_factor
            )
            for s in self.samplers
        ])
        return ys

    def dataset_size(self):
        return min(s.dataset_size() for s in self.samplers)

    def sample(self, n_samples=1, n_jobs=1):
        m = self.dataset_size()

        # generate random index
        idx = np.arange(m)
        np.random.shuffle(idx)
        idx = np.hstack((
            np.arange(m).repeat(int(math.ceil(n_samples // m))).flatten(),
            idx[:n_samples%m]
        ))
        np.random.shuffle(idx)

        # load
        if n_jobs <= 0:
            n_jobs = os.cpu_count()
        if n_jobs > 1:
            p = multiprocessing.Pool(n_jobs, maxtasksperchild=1)
            samples = np.hstack(list(p.map(self.load, idx)))
            p.close()
            p.join()
        else:
            samples = np.hstack(list(map(self.load, idx)))
        return samples.transpose((1, 0, 2))