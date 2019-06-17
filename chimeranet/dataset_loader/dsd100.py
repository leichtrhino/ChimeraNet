
import os
import io
import zipfile
import librosa
import soundfile

from ..audio_loader import AudioLoader

class DSD100Loader(AudioLoader):
    # inst_list contains ('bass', 'drums', 'other', 'vocals')
    def __init__(self, path, inst_list, validation=False):
        super().__init__()
        self.path = path
        self.inst_list = inst_list
        prefix = 'DSD100/Sources/{}/'.format(
            'Dev' if not validation else 'Test'
        )
        self.zf = zipfile.ZipFile(path)
        self.name_list = list(
            i.filename for i in self.zf.infolist()
            if i.filename.startswith(prefix)\
                and i.filename != prefix and i.is_dir()
        )
    def __del__(self):
        self.zf.close()
    def load_audio(self, index, sr):
        data = None
        for inst in self.inst_list:
            b = self.zf.read(self.name_list[index]+inst+'.wav')
            if data is None:
                data, samplerate = soundfile.read(io.BytesIO(b))
            else:
                data_, samplerate = soundfile.read(io.BytesIO(b))
                data += data_
        return librosa.to_mono(librosa.resample(data.T, samplerate, sr))
    def __len__(self):
        return len(self.name_list)

class DSD100MelodyLoader(DSD100Loader):
    def __init__(self, path, validation=False):
        super().__init__(path, ['bass', 'drums', 'other'], validation)

class DSD100VocalLoader(DSD100Loader):
    def __init__(self, path, validation=False):
        super().__init__(path, ['vocals'], validation)
