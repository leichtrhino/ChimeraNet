
import os
import io
import tarfile
import zipfile
import librosa
import soundfile

class AudioLoader:
    def load_audio(self, index, sr):
        return None
    def __len__(self):
        return 0

class DirAudioLoader(AudioLoader):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.sorted_file_list = sorted(os.listdir(path))
    def load_audio(self, index, sr):
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
        with zipfile.ZipFile(zippath) as zf:
            self.name_list = list(
                i.filename for i in zf.infolist()
                if i.filename.startswith(path) and not i.is_dir()
            )
    def load_audio(self, index, sr):
        with zipfile.ZipFile(self.zippath) as zf:
            b = zf.read(self.name_list[index])
            data, samplerate = soundfile.read(io.BytesIO(b))
        data = data.T
        return librosa.to_mono(librosa.resample(data, samplerate, sr))
    def __len__(self):
        return len(self.name_list)

class TarAudioLoader(AudioLoader):
    def __init__(self, tarpath, path):
        super().__init__()
        self.tarpath = tarpath
        self.path = path
        with tarfile.open(tarpath) as tf:
            self.name_list = list(
                i.name for i in tf.getmembers()
                if i.name.startswith(path) and i.isfile()
            )
    def load_audio(self, index, sr):
        with tarfile.open(self.tarpath) as tf:
            with tf.extractfile(self.name_list[index]) as f:
                data, samplerate = soundfile.read(f)
        data = data.T
        return librosa.to_mono(librosa.resample(data, samplerate, sr))
    def __len__(self):
        return len(self.name_list)
