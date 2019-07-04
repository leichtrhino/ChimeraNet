
import os
import io
import zipfile
import librosa
import soundfile

from ..audio_loader import AudioLoader

class DSD100Loader(AudioLoader):
    # inst_list contains ('bass', 'drums', 'other', 'vocals')
    def __init__(self, path, inst_list, dev=True, test=False):
        super().__init__()
        self.path = path
        self.inst_list = inst_list
        prefix_list = []
        if dev:
            prefix_list.append('DSD100/Sources/Dev/')
        if test:
            prefix_list.append('DSD100/Sources/Test/')
        zf = zipfile.ZipFile(path)
        self.name_list = list(
            i.filename for i in zf.infolist()
            if any(
                i.filename.startswith(prefix)\
                and i.filename != prefix and i.is_dir()
                for prefix in prefix_list
            )
        )
        zf.close()
    def __del__(self):
        super().__del__()
    def _load_audio(self, index, sr):
        zf = zipfile.ZipFile(self.path)
        data = None
        for inst in self.inst_list:
            b = zf.read(self.name_list[index]+inst+'.wav')
            if data is None:
                data, samplerate = soundfile.read(io.BytesIO(b))
            else:
                data_, samplerate = soundfile.read(io.BytesIO(b))
                data += data_
        data /= len(self.inst_list)
        zf.close()
        return librosa.resample(librosa.to_mono(data.T), samplerate, sr)
    def __len__(self):
        return len(self.name_list)

class DSD100MelodyLoader(DSD100Loader):
    def __init__(self, path, dev=True, test=False):
        super().__init__(path, ['bass', 'drums', 'other'], dev, test)

class DSD100VocalLoader(DSD100Loader):
    def __init__(self, path, dev=True, test=False):
        super().__init__(path, ['vocals'], dev, test)
