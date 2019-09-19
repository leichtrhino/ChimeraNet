
import io
import zipfile

from .base import Dataset, load

class VoxCeleb(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        zf = zipfile.ZipFile(path)
        self.name_list = sorted([
            i.filename for i in zf.infolist() if not i.is_dir()
        ])
        zf.close()
    def load(self, index, sr, offset=0., duration=None):
        name = self.name_list[index]
        zf = zipfile.ZipFile(self.path)
        y = load(io.BytesIO(zf.read(name)), sr, offset, duration)
        zf.close()
        return y
    def __len__(self):
        return len(self.name_list)
