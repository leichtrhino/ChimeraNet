#!/usr/env/bin python
import os, sys
import numpy as np
import pickle
import h5py
import librosa

sys.path.append(os.path.join(os.path.split(__file__)[0], '..'))
from chimeranet.model import ChimeraNetModel
from chimeranet.audio_sampler import SpectrogramSampler
from chimeranet.dataset_loader.dsd100\
    import DSD100MelodyLoader, DSD100VocalLoader
from chimeranet.dataset_loader.esc50 import ESC50Loader
from chimeranet.dataset_loader.voxceleb import VoxCelebLoader
from chimeranet.preprocessing import to_mixture, to_true_pair

def main():
    # parameters
    time = 0.5
    sr, n_fft, hop_length, n_mels = 16000, 512, 128, 64
    dataset_path = 'dataset_trisep.h5'
    model_path = 'model_trisep.h5'
    history_path = 'history_trisep.json.pkl'

    vox_dev_path = 'vox1_dev_wav.zip'
    vox_test_path = 'vox1_test_wav.zip'
    dsd_path = 'DSD100.zip'
    esc_path = 'ESC-50-master.zip'
    esc_cat_list = ESC50Loader.all_category_list(esc_path)
    esc_dev_fold = (1, 2, 3, 4)
    esc_test_fold = (5,)

    # build spec. sampler for training and validation
    ss = SpectrogramSampler()
    ss.add_reader(VoxCelebLoader(dev_path=vox_dev_path))
    ss.add_reader(DSD100MelodyLoader(dsd_path))
    ss.add_reader(ESC50Loader(esc_path, category_list=esc_cat_list, fold=esc_dev_fold))
    ss.time(time).n_mels(n_mels).sr(sr).n_fft(n_fft).hop_length(hop_length)
    ss.augment_time_range(2/3, 4/3).augment_freq_range(-0.5, 0.5).augment_amp_range(-10, 10)
    ss.sync_flag(False)

    ss_val = SpectrogramSampler()
    ss_val.add_reader(VoxCelebLoader(dev_path=vox_test_path))
    ss_val.add_reader(DSD100MelodyLoader(dsd_path, test=True))
    ss_val.add_reader(ESC50Loader(esc_path, category_list=esc_cat_list, fold=esc_test_fold))
    ss_val.time(time).n_mels(n_mels).sr(sr).n_fft(n_fft).hop_length(hop_length)
    ss_val.sync_flag(False)

    # build model
    cm = ChimeraNetModel(ss.get_frames(), n_mels, ss.get_number_of_channels(), 20)
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

    # obtain training data from spectrogram sampler and train model
    sample_size, batch_size = 32000, 32
    specs_train = ss.make_specs(sample_size, 50)
    specs_valid = ss_val.make_specs(sample_size // 100, 50)
    with h5py.File(dataset_path, 'w') as f:
        f.create_dataset('specs_train', data=specs_train)
        f.create_dataset('specs_valid', data=specs_valid)
    x_train = to_mixture(specs_train)
    y_train = to_true_pair(specs_train)
    x_valid = to_mixture(specs_valid)
    y_valid = to_true_pair(specs_valid)
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=30,
        validation_data=(x_valid, y_valid)
    )
    model.save(model_path)
    with open(history_path, 'wb') as fp:
        pickle.dump(history.history, fp)

if __name__ == '__main__':
    main()
