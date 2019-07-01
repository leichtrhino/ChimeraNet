
import os
import io
import zipfile
import librosa
import soundfile

from ..audio_loader import AudioLoader

class ESC50Loader(AudioLoader):
    def __init__(self, path, category_list, fold=(1, 2, 3, 4)):
        super().__init__()
        name_col_idx, fold_col_idx, category_col_idx = 0, 1, 3
        self.zf = zipfile.ZipFile(path)
        rows = self.zf.read('ESC-50-master/meta/esc50.csv')\
            .decode().splitlines()[1:]
        rows = [row.split(',') for row in rows]
        check_category = lambda r:\
            any(r[category_col_idx] == cat for cat in category_list)
        check_fold = lambda r: int(r[fold_col_idx]) in fold
        self.name_list = [
            row[name_col_idx] for row in rows
            if check_category(row) and check_fold(row)
        ]
    def __del__(self):
        self.zf.close()
    def _load_audio(self, index, sr):
        b = self.zf.read('ESC-50-master/audio/'+self.name_list[index])
        data, samplerate = soundfile.read(io.BytesIO(b))
        return librosa.resample(librosa.to_mono(data.T), samplerate, sr)
    def __len__(self):
        return len(self.name_list)
    @staticmethod
    def all_category_list(path):
        target_col_idx, category_col_idx = 2, 3
        zf = zipfile.ZipFile(path)
        rows = zf.read('ESC-50-master/meta/esc50.csv')\
            .decode().splitlines()[1:]
        rows = [row.split(',') for row in rows]
        cats = sorted(
            set((r[target_col_idx], r[category_col_idx]) for r in rows)
        )
        zf.close()
        return [cat[1] for cat in cats]
