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
from chimeranet.preprocessing import to_mixture, to_true_pair

def main():
    # parameters
    time = 0.5
    sr, n_fft, hop_length, n_mels = 16000, 512, 128, 64
    dataset_path = 'dataset_dsd100.h5'
    model_path = 'model_dsd100.h5'
    history_path = 'history_dsd100.json.pkl'

    # build spec. sampler for training and validation
    ss = SpectrogramSampler()
    ss.add_reader(DSD100VocalLoader('DSD100.zip'))
    ss.add_reader(DSD100MelodyLoader('DSD100.zip'))
    ss.time(time).n_mels(n_mels).sr(sr).n_fft(n_fft).hop_length(hop_length)
    ss.augment_time_range(2/3, 4/3).augment_freq_range(-0.5, 0.5).augment_amp_range(-10, 10)
    ss.sync_flag(True)

    ss_val = SpectrogramSampler()
    ss_val.add_reader(DSD100VocalLoader('DSD100.zip', validation=True))
    ss_val.add_reader(DSD100MelodyLoader('DSD100.zip', validation=True))    
    ss_val.time(time).n_mels(n_mels).sr(sr).n_fft(n_fft).hop_length(hop_length)
    ss_val.sync_flag(True)
    
    # build model
    cm = ChimeraNetModel(
        ss.get_frames(), n_mels, ss.get_number_of_channels(), 20
    )
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
    sample_size, batch_size = 16000, 32
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
        epochs=200,
        validation_data=(x_valid, y_valid)
    )
    model.save(model_path)
    with open(history_path, 'wb') as fp:
        pickle.dump(history.history, fp)

if __name__ == '__main__':
    main()
