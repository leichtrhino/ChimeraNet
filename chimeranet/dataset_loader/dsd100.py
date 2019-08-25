
import os
import io
import zipfile
import tempfile
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
        self.name_list = sorted(
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
    def _load_audio(self, index, sr, offset=0., duration=None):
        zf = zipfile.ZipFile(self.path)
        data = None
        offset_ = -1
        for inst in self.inst_list:
            name = self.name_list[index]+inst+'.wav'
            data_, sr_ = soundfile.read(io.BytesIO(zf.read(name)))
            if duration is not None:
                sample_size = int(duration * sr_)
                sample_offset = int(offset * max(data_.shape[0] - sample_size, 0))
                data_ = data_[sample_offset:sample_offset+sample_size]
            data_ = librosa.resample(librosa.to_mono(data_.T), sr_, sr)
            if data is None:
                data = data_
            else:
                data += data_
        zf.close()
        data /= len(self.inst_list)
        return data
    def __len__(self):
        return len(self.name_list)

class DSD100MelodyLoader(DSD100Loader):
    def __init__(self, path, dev=True, test=False):
        super().__init__(path, ['bass', 'drums', 'other'], dev, test)

class DSD100VocalLoader(DSD100Loader):
    def __init__(self, path, dev=True, test=False):
        super().__init__(path, ['vocals'], dev, test)
