
import os
import io
import tarfile
import zipfile
import tempfile
import numpy as np
import librosa
import soundfile

_supported_extensions = set(('aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav'))
def is_audio_file(path):
    return os.path.splitext(path)[1][1:] in _supported_extensions

"""
wrapper function of librosa.core.load
"""
def load(path, sr=44100, offset=0., duration=None):
    if isinstance(path, io.IOBase):
        data, sr_ = soundfile.read(path, dtype='float32')
        if duration is not None:
            sample_size = int(duration * sr_)
            sample_offset = int(offset * max(data.shape[0] - sample_size, 0))
            data = data[sample_offset:sample_offset+sample_size]
        else:
            sample_offset = int(offset * max(data.shape[0], 0))
            data = data[sample_offset:]
        data = librosa.resample(librosa.to_mono(data.T), sr_, sr)
        return data
    else:
        if duration is not None:
            offset = offset * (librosa.get_duration(filename=path) - duration)
        else:
            offset = offset * librosa.get_duration(filename=path)
        return librosa.load(path, sr=sr, offset=offset, duration=duration)[0]

class Dataset:
    def load(self, index, sr, offset=0., duration=None):
        return None
    def __len__(self):
        return 0
    def __getitem__(self, key):
        parent = self
        if type(key) == int:
            p_index = [key]
        elif type(key) == tuple or type(key) == list\
            or type(key) == np.ndarray:
            p_index = key
        elif type(key) == slice:
            p_index = list(range(*key.indices(len(parent))))
        else:
            raise TypeError(type(key), 'not supported')
        return _SubDataset(self, p_index)
    def __add__(self, dataset):
        if not isinstance(dataset, Dataset):
            raise TypeError(type(dataset), 'not supported')
        return _UnionDataset(self, dataset)

class _UnionDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.idx_to_idx = np.cumsum([len(d) for d in datasets])
    def load(self, index, sr, offset=0, duration=None):
        idx = np.searchsorted(self.idx_to_idx, index, 'right')
        midx = index - (0 if idx == 0 else self.idx_to_idx[idx-1])
        return self.datasets[idx].load(
            midx, sr, offset=offset, duration=duration)
    def __len__(self):
        return self.idx_to_idx[-1]

class _SubDataset(Dataset):
    def __init__(self, parent, p_index):
        self.parent = parent
        self.p_index = p_index
    def load(self, index, sr, offset=0, duration=None):
        return self.parent.load(
            self.p_index[index], sr, offset=offset, duration=duration)
    def __len__(self):
        return len(self.p_index)

class DirDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.file_list = sorted(sum(
            (
                [os.path.join(dp, f) for f in fn if is_audio_file(f)]
                for dp, dn, fn in os.walk(path)
            ),
            []
        ))
    def load(self, index, sr, offset=0., duration=None):
        return load(self.file_list[index], sr, offset, duration)
    def __len__(self):
        return len(self.file_list)

class ZipDataset(Dataset):
    def __init__(self, zippath, path=''):
        super().__init__()
        self.zippath = zippath
        self.path = path
        zf = zipfile.ZipFile(zippath)
        self.name_list = sorted(
            i.filename for i in zf.infolist()
            if i.filename.startswith(path) and not i.is_dir()
            and is_audio_file(i.filename)
        )
        zf.close()
    def load(self, index, sr, offset=0., duration=None):
        name = self.name_list[index]
        zf = zipfile.ZipFile(self.zippath)
        y = load(io.BytesIO(zf.read(name)), sr, offset, duration)
        zf.close()
        return y
    def __len__(self):
        return len(self.name_list)

class TarDataset(Dataset):
    def __init__(self, tarpath, path=''):
        super().__init__()
        self.tarpath = tarpath
        self.path = path
        tf = tarfile.open(tarpath)
        self.name_list = sorted(
            i.name for i in tf.getmembers()
            if i.name.startswith(path) and i.isfile()
            and is_audio_file(i.name)
        )
        tf.close()
    def load(self, index, sr, offset=0., duration=None):
        name = self.name_list[index]
        tf = tarfile.open(self.tarpath)
        y = load(tf.extractfile(name), sr, offset, duration)
        tf.close()
        return y
    def __len__(self):
        return len(self.name_list)
