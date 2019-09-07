
import io
import zipfile
import numpy as np

from .base import Dataset, load

class DSD100(Dataset):
    # inst_list contains ('bass', 'drums', 'other', 'vocals')
    def __init__(self, path, inst_list, dev=True, test=False):
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
    def load(self, index, sr, offset=0., duration=None):
        zf = zipfile.ZipFile(self.path)
        y = np.array([
            load(
                io.BytesIO(zf.read(self.name_list[index]+inst+'.wav')),
                sr, offset, duration
            ) for inst in self.inst_list
        ]).mean(axis=0)
        zf.close()
        return y
    def __len__(self):
        return len(self.name_list)

class DSD100Melody(DSD100):
    def __init__(self, path, dev=True, test=False):
        super().__init__(path, ['bass', 'drums', 'other'], dev, test)

class DSD100Vocal(DSD100):
    def __init__(self, path, dev=True, test=False):
        super().__init__(path, ['vocals'], dev, test)
