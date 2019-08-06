
import os
import io
import tarfile
import zipfile
import tempfile
import numpy as np
import librosa
import soundfile

class AudioLoader:
    def __init__(self):
        pass
    def __del__(self):
        pass
    def load_audio(self, index, sr, offset=0., duration=None):
        return self._load_audio(index, sr, offset=offset, duration=duration)
    def _load_audio(self, index, sr, offset=0., duration=None):
        return None
    def __len__(self):
        return 0
    def __getitem__(self, key):
        parent = self
        if type(key) == int:
            p_index = [key]
        elif type(key) == tuple:
            p_index = key
        elif type(key) == slice:
            p_index = list(range(*key.indices(len(parent))))
        elif type(key) == np.ndarray:
            p_index = key
        else:
            raise TypeError(type(key), 'not supported')
        class _MiniLoader(AudioLoader):
            def _load_audio(self, index, sr, offset=0, duration=None):
                return parent._load_audio(
                    p_index[index], sr, offset=offset, duration=duration)
            def __len__(self):
                return len(p_index)
        return _MiniLoader()

class FakeAudioLoader(AudioLoader):
    def __init__(self, n_samples, audio_size):
        super().__init__()
        self.n_samples = n_samples
        self.audio_size = audio_size
    def _load_audio(self, index, sr, offset=0, duration=None):
        return np.full(self.audio_size, index).astype(float)
    def __len__(self):
        return self.n_samples

class Combiner(AudioLoader):
    def __init__(self, *loaders):
        super().__init__()
        self.loaders = loaders
        self.idx_to_idx = np.cumsum([len(l) for l in loaders])
    def _load_audio(self, index, sr, offset=0, duration=None):
        idx = np.searchsorted(self.idx_to_idx, index, 'right')
        midx = index - (0 if idx == 0 else self.idx_to_idx[idx-1])
        return self.loaders[idx].load_audio(
            midx, sr, offset=offset, duration=duration)
    def __len__(self):
        return self.idx_to_idx[-1]

def split_loader(loader, weights, shuffle=True):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    # TODO check dims of weights
    ranges = np.hstack(((0,), np.cumsum(weights*len(loader)))).astype(int)
    index = np.arange(len(loader))
    if shuffle:
        np.random.shuffle(index)
    return [loader[index[a:b]] for a, b in zip(ranges[:-1], ranges[1:])]

class DirAudioLoader(AudioLoader):
    def __init__(self, path):
        super().__init__()
        self.path = path
        ext = set(['aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav'])
        self.sorted_file_list = sorted(sum(
            ([os.path.join(dp, f)
                for f in fn if os.path.splitext(f)[1][1:] in ext]
                for dp, dn, fn in os.walk(path)),
            []
        ))
    def _load_audio(self, index, sr, offset=0., duration=None):
        if duration is not None:
            offset *= librosa.core.get_duration(filename=ifn) - duration
        y, _ = librosa.core.load(
            self.sorted_file_list[index], sr=sr, offset=offset, duration=duration)
        return y
    def __len__(self):
        return len(self.sorted_file_list)

class ZipAudioLoader(AudioLoader):
    def __init__(self, zippath, path=''):
        super().__init__()
        self.zippath = zippath
        self.path = path
        ext = set(['aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav'])
        zf = zipfile.ZipFile(zippath)
        self.name_list = sorted(list(
            i.filename for i in zf.infolist()
            if i.filename.startswith(path) and not i.is_dir()
            and os.path.splitext(i.filename)[1][1:] in ext
        ))
        zf.close()
    def __del__(self):
        super().__del__()
    def _load_audio(self, index, sr, offset=0., duration=None):
        name = self.name_list[index]
        f = os.path.splitext(name)[1]
        zf = zipfile.ZipFile(self.zippath)
        with tempfile.TemporaryDirectory() as dirname:
            ifn = os.path.join(dirname, 'in{}'.format(f))
            with open(ifn, 'wb') as fp:
                fp.write(zf.read(name))
            if duration is not None:
                offset *= librosa.core.get_duration(filename=ifn) - duration
            data, _ = librosa.load(
                ifn, sr=sr, offset=offset, duration=duration)
        zf.close()
        return data
    def __len__(self):
        return len(self.name_list)

class TarAudioLoader(AudioLoader):
    def __init__(self, tarpath, path=''):
        super().__init__()
        self.tarpath = tarpath
        self.path = path
        ext = set(['aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav'])
        tf = tarfile.open(tarpath)
        self.name_list = sorted(list(
            i.name for i in tf.getmembers()
            if i.name.startswith(path) and i.isfile()
            and os.path.splitext(i.name)[1][1:] in ext
        ))
        tf.close()
    def __del__(self):
        super().__del__()
    def load_audio(self, index, sr, offset=0., duration=None):
        name = self.name_list[index]
        f = os.path.splitext(name)[1]
        tf = tarfile.open(self.tarpath)
        with tempfile.TemporaryDirectory() as dirname:
            ifn = os.path.join(dirname, 'in{}'.format(f))
            with open(ifn, 'wb') as fp1:
                with tf.extractfile(name) as fp2:
                    fp1.write(fp2.read())
            if duration is not None:
                offset *= librosa.core.get_duration(filename=ifn) - duration
            data, _ = librosa.load(
                ifn, sr=sr, offset=offset, duration=duration)
        tf.close()
        return data
    def __len__(self):
        return len(self.name_list)
