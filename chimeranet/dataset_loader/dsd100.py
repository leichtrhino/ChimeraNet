
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
    def _load_audio(self, index, sr, offset=0., duration=None):
        zf = zipfile.ZipFile(self.path)
        data = None
        _offset = -1
        with tempfile.TemporaryDirectory() as dirname:
            ifn = os.path.join(dirname, 'in.wav')
            for inst in self.inst_list:
                name = self.name_list[index]+inst+'.wav'
                with open(ifn, 'wb') as fp:
                    fp.write(zf.read(name))
                if duration is not None and _offset < 0:
                    _offset = offset * librosa.core.get_duration(filename=ifn)\
                        - duration
                data_, _ = librosa.load(
                    ifn, sr=sr, offset=_offset, duration=duration)
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
