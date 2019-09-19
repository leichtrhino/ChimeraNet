
import os
import io
import tarfile
import tempfile
import librosa
import subprocess

import io
import zipfile

from .base import Dataset, load

class LMD(Dataset):
    # fold = 'def' to validation fold
    def __init__(self, path=None, root='lmd_full', fold='0123456789abc'):
        self.root = root
        self.path = path
        tf = tarfile.open(path)
        self.name_list =  sorted([
            i.name for i in tf.getmembers()
            if i.isfile() and\
                any(i.name.startswith('{}/{}/'.format(root, p)) for p in fold)
        ])
        tf.close()
    def __len__(self):
        return len(self.name_list)
    def load(self, index, sr, offset=0., duration=None):
        name = self.name_list[index]
        tf = tarfile.open(self.path)
        with tempfile.TemporaryDirectory() as dirname:
            ifn = os.path.join(dirname, 'in.mid')
            ofn = os.path.join(dirname, 'out.wav')
            with open(ifn, 'wb') as fp:
                f = tf.extractfile(name)
                fp.write(f.read())
                f.close()
            cmd = ['musescore', ifn, '-o', ofn]
            proc = subprocess.run(cmd, stderr=subprocess.PIPE)
            assert proc.returncode == 0,\
            'musescore exit with return code {} and stderr:\n{}'\
            .format(proc.returncode, proc.stderr.decode())
            data = load(ofn, sr, offset, duration)
        tf.close()
        return data
