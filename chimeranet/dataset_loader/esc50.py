
import os
import io
import zipfile
import librosa
import soundfile

from ..audio_loader import AudioLoader

class ESC50Loader(AudioLoader):
    def __init__(self, path, category_list, fold=(1, 2, 3, 4), cache=True):
        super().__init__(cache)
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
        super().__del__()
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
    @staticmethod
    def major_categories():
        return ['animals', 'natural', 'human', 'interior', 'exterior']
    @staticmethod
    def categories_of(major_categories):
        categories = {
            'animals': ['dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow'],
            'natural': ['rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds', 'water_drops', 'wind', 'pouring_water', 'toilet_flush', 'thunderstorm'],
            'human': ['crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing', 'footsteps', 'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping'],
            'interior': ['door_wood_knock', 'mouse_click', 'keyboard_typing', 'door_wood_creaks', 'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'clock_tick', 'glass_breaking'],
            'exterior': ['helicopter', 'chainsaw', 'siren', 'car_horn', 'engine', 'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw']
        }
        if type(major_categories) == str:
            major_categories = [major_categories]
        return sum((categories[c] for c in major_categories), [])
    @staticmethod
    def all_categories():
        return ESC50Loader.categories_of(ESC50Loader.major_categories())
