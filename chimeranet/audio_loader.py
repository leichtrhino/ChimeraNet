
import os
import librosa

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
    def load_audio(self, index, sr):
        pass
    def __len__(self):
        pass

class TarAudioLoader(AudioLoader):
    def __init__(self, tarpath, path):
        super().__init__()
        self.tarpath = tarpath
        self.path = path
    def load_audio(self, index, sr):
        pass
    def __len__(self):
        pass
