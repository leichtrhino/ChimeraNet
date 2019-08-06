
import os
import io
import zipfile
import tempfile
import librosa
import soundfile

from ..audio_loader import AudioLoader

class VoxCelebLoader(AudioLoader):
    def __init__(self, path):
        super().__init__()
        self.path = path
        zf = zipfile.ZipFile(path)
        self.name_list =  [
            i.filename for i in zf.infolist() if not i.is_dir()
        ]
        zf.close()
    def _load_audio(self, index, sr, offset=0., duration=None):
        name = self.name_list[index]
        f = os.path.splitext(name)[1][1:]
        zf = zipfile.ZipFile(self.path)
        with tempfile.TemporaryDirectory() as dirname:
            ifn = os.path.join(dirname, 'in.{}'.format(f))
            with open(ifn, 'wb') as fp:
                fp.write(zf.read(name))
            if duration is not None:
                offset *= librosa.core.get_duration(filename=ifn) - duration
            data, _ = librosa.load(
                ifn, sr=sr, offset=offset, duration=duration)
        zf.close()
        return data
    def __len__(self):
        return len(self.name_list)
