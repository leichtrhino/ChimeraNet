
import os
import io
import tarfile
import zipfile
import numpy as np
import librosa
import soundfile

class AudioLoader:
    def __init__(self):
        pass
    def __del__(self):
        pass
    def load_audio(self, index, sr):
        return self._load_audio(index, sr)
    def _load_audio(self, index, sr):
        return None
    def __len__(self):
        return 0

class FakeAudioLoader(AudioLoader):
    def __init__(self, n_samples, audio_size):
        super().__init__()
        self.n_samples = n_samples
        self.audio_size = audio_size
    def _load_audio(self, index, sr):
        return np.ones(self.audio_size)
    def __len__(self):
        return self.n_samples

class DirAudioLoader(AudioLoader):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.sorted_file_list = sorted(os.listdir(path))
    def _load_audio(self, index, sr):
        y, _ = librosa.core.load(
            os.path.join(self.path, self.sorted_file_list[index]),
            sr=sr
        )
        return y
    def __len__(self):
        return len(self.sorted_file_list)

class ZipAudioLoader(AudioLoader):
    def __init__(self, zippath, path):
        super().__init__()
        self.zippath = zippath
        self.path = path
        zf = zipfile.ZipFile(zippath)
        self.name_list = list(
            i.filename for i in zf.infolist()
            if i.filename.startswith(path) and not i.is_dir()
        )
        zf.close()
    def __del__(self):
        super().__del__()
    def _load_audio(self, index, sr):
        zf = zipfile.ZipFile(self.zippath)
        b = zf.read(self.name_list[index])
        data, samplerate = soundfile.read(io.BytesIO(b))
        zf.close()
        return librosa.resample(librosa.to_mono(data.T), samplerate, sr)
    def __len__(self):
        return len(self.name_list)

class TarAudioLoader(AudioLoader):
    def __init__(self, tarpath, path):
        super().__init__()
        self.tarpath = tarpath
        self.path = path
        tf = tarfile.open(tarpath)
        self.name_list = list(
            i.name for i in self.tf.getmembers()
            if i.name.startswith(path) and i.isfile()
        )
        tf.close()
    def __del__(self):
        super().__del__()
    def load_audio(self, index, sr):
        tf = tarfile.open(self.tarpath)
        with self.tf.extractfile(self.name_list[index]) as f:
            data, samplerate = soundfile.read(f)
        tf.close()
        return librosa.resample(librosa.to_mono(data.T), samplerate, sr)
    def __len__(self):
        return len(self.name_list)
