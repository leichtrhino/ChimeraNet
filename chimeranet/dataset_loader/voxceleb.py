
import os
import io
import zipfile
import librosa
import soundfile

from ..audio_loader import AudioLoader

class VoxCelebLoader(AudioLoader):
    def __init__(self, dev_path=None, test_path=None):
        super().__init__()
        self.dev_path, self.test_path = dev_path, test_path
        self.name_list = []
        if dev_path:
            dev_zf = zipfile.ZipFile(dev_path)
            self.name_list +=  [
                (i.filename, 'dev')
                for i in dev_zf.infolist() if not i.is_dir()
            ]
            dev_zf.close()
        if test_path:
            test_zf = zipfile.ZipFile(test_path)
            self.name_list += [
                (i.filename, 'test')
                for i in test_zf.infolist() if not i.is_dir()
            ]
            test_zf.close()
    def __del__(self):
        super().__del__()
    def _load_audio(self, index, sr):
        name, arc = self.name_list[index]
        zf = zipfile.ZipFile(self.dev_path if arc == 'dev' else self.test_path)
        data, samplerate = soundfile.read(io.BytesIO(zf.read(name)))
        zf.close()
        return librosa.resample(librosa.to_mono(data.T), samplerate, sr)
    def __len__(self):
        return len(self.name_list)
