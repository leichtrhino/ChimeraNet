#!/usr/env/bin python
import os, sys
import importlib.util
import numpy as np
import pickle
import h5py
import librosa

if __name__ == '__mp_main__': # suppress import message on worker
    sys.stdout = sys.stderr = open(os.devnull, 'w')

if not importlib.util.find_spec('chimeranet'):
    print('ChimeraNet is not installed, import from source.')
    sys.path.append(os.path.join(os.path.split(__file__)[0], '..'))
else:
    print('Using installed ChimeraNet.')

from chimeranet.model import ChimeraNetModel
from chimeranet.audio_mixer import AudioMixer
from chimeranet.dataset_loader.dsd100\
    import DSD100MelodyLoader, DSD100VocalLoader
from chimeranet.dataset_loader.esc50 import ESC50Loader
from chimeranet.dataset_loader.voxceleb import VoxCelebLoader
from chimeranet.preprocessing import to_mixture, to_true_pair

def main():
    # parameters
    time = 0.75
    sr, n_fft, hop_length, n_mels = 16000, 512, 128, 96
    dataset_path = 'dataset_trisep.h5'
    model_path = 'model_trisep.h5'
    history_path = 'history_trisep.json.pkl'

    vox_dev_path = 'vox1_dev_wav.zip'
    vox_test_path = 'vox1_test_wav.zip'
    dsd_path = 'DSD100.zip'
    esc_path = 'ESC-50-master.zip'
    esc_cat_list = ESC50Loader.all_categories()
    esc_dev_fold = (1, 2, 3, 4)
    esc_test_fold = (5,)

    # build spec. sampler for training and validation
    am = AudioMixer()
    am.add_loader(
        VoxCelebLoader(dev_path=vox_dev_path),
        a_time=(-1, 1), a_freq=(-1, 1), a_amp=(-5, 5)
    )
    am.add_loader(
        DSD100MelodyLoader(dsd_path),
        a_time=(-1, 1), a_freq=(-1, 1), a_amp=(-5, 5)
    )
    am.add_loader(
        ESC50Loader(esc_path, category_list=esc_cat_list, fold=esc_dev_fold),
        a_time=(-0.3, 0.3), a_freq=(-0.2, 0.2), a_amp=(-10, 0)
    )
    am.time(time).n_mels(n_mels).sr(sr).n_fft(n_fft).hop_length(hop_length)
    am.sync_flag(False)

    am_val = AudioMixer()
    am_val.add_loader(
        VoxCelebLoader(dev_path=vox_test_path)
    )
    am_val.add_loader(
        DSD100MelodyLoader(dsd_path, test=True)
    )
    am_val.add_loader(
        ESC50Loader(esc_path, category_list=esc_cat_list, fold=esc_test_fold)
    )
    am_val.time(time).n_mels(n_mels).sr(sr).n_fft(n_fft).hop_length(hop_length)
    am_val.sync_flag(False)

    T, F, C, D = am.get_frames(), n_mels, am.get_number_of_channels(), 20

    # obtain training data from spectrogram sampler and train model
    sample_size, batch_size = 64000, 32
    specs_train = am.make_specs(sample_size, n_jobs=16)
    specs_valid = am_val.make_specs(sample_size // 10, n_jobs=16)
    with h5py.File(dataset_path, 'w') as f:
        f.create_dataset('specs_train', data=specs_train)
        f.create_dataset('specs_valid', data=specs_valid)
    x_train = to_mixture(specs_train)
    y_train = to_true_pair(specs_train)
    x_valid = to_mixture(specs_valid)
    y_valid = to_true_pair(specs_valid)

    # build model
    cm = ChimeraNetModel(T, F, C, D)
    model = cm.build_model()
    model.compile(
        'rmsprop',
        loss={
            'embedding': cm.loss_deepclustering(),
            'mask': cm.loss_mask()
        },
        loss_weights={
            'embedding': 0.5,
            'mask': 0.5
        }
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=10,
        validation_data=(x_valid, y_valid)
    )
    model.save(model_path)
    with open(history_path, 'wb') as fp:
        pickle.dump(history.history, fp)

if __name__ == '__main__':
    main()
