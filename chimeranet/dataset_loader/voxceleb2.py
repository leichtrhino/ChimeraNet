
import os
import io
import zipfile
import tempfile
import librosa
import subprocess

from .voxceleb import VoxCelebLoader

class VoxCeleb2Loader(VoxCelebLoader):
    def __init__(self, dev_path=None, test_path=None, cache=True):
        super().__init__(dev_path, test_path, cache)
    def _load_audio(self, index, sr):
        name, arc = self.name_list[index]
        zf = self.dev_zf if arc == 'dev' else self.test_zf
        with tempfile.TemporaryDirectory() as dirname:
            ifn = os.path.join(dirname, 'in.m4a')
            ofn = os.path.join(dirname, 'out.m4a')
            with open(ifn, 'wb') as fp:
                fp.write(zf.read(name))
            cmd = ['ffmpeg', '-i', ifn, ofn]
            proc = subprocess.run(cmd, stderr=subprocess.PIPE)
            assert proc.returncode == 0,\
            'ffmpeg exit with return code {} and stderr:\n{}'\
            .format(proc.returncode, proc.stderr.decode())
            data, _ = librosa.load(ofn, sr=sr)
        return data
