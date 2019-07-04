
import os
import io
import zipfile
import librosa
import soundfile
from itertools import chain

from ..audio_loader import AudioLoader

class VoxCelebLoader(AudioLoader):
    def __init__(self, dev_path=None, test_path=None, cache=True):
        super().__init__(cache)
        self.dev_zf, self.test_zf = None, None
        namelistiter = []
        if dev_path:
            self.dev_zf = zipfile.ZipFile(dev_path)
            namelistiter = chain(
                iter(
                    (i.filename, 'dev')
                    for i in self.dev_zf.infolist() if not i.is_dir()
                ),
                namelistiter
            )
        if test_path:
            self.test_zf = zipfile.ZipFile(test_path)
            namelistiter = chain(
                iter(
                    (i.filename, 'test')
                    for i in self.test_zf.infolist() if not i.is_dir()
                ),
                namelistiter
            )
        self.name_list = list(namelistiter)
    def __del__(self):
        if self.dev_zf:
            self.dev_zf.close()
        if self.test_zf:
            self.test_zf.close()
        super().__del__()
    def _load_audio(self, index, sr):
        name, arc = self.name_list[index]
        zf = self.dev_zf if arc == 'dev' else self.test_zf
        data, samplerate = soundfile.read(io.BytesIO(zf.read(name)))
        return librosa.resample(librosa.to_mono(data.T), samplerate, sr)
    def __len__(self):
        return len(self.name_list)
