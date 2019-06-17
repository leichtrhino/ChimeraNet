#!/usr/env/bin python
import os, sys
import numpy as np
import h5py
import librosa

sys.path.append(os.path.join(os.path.split(__file__)[0], '..'))
import chimeranet.model
from chimeranet.audio_sampler import SpectrogramSampler
from chimeranet.dataset_loader.dsd100\
    import DSD100MelodyLoader, DSD100VocalLoader
from chimeranet.preprocessing import to_mixture, to_true_pair

def main():
    # build spectrogram sampler
    time = 0.5
    sr, n_fft, hop_length, n_mels = 16000, 512, 128, 64
    dataset_path = 'dataset_dsd100.h5'
    model_path = 'model_dsd100.h5'
    ss = SpectrogramSampler()
    ss.add_reader(DSD100VocalLoader('DSD100.zip'))
    ss.add_reader(DSD100MelodyLoader('DSD100.zip'))
    ss.time(time).n_mels(n_mels).sr(sr).n_fft(n_fft).hop_length(hop_length)
    ss.augment_amp_range(-5, 5)
    ss.augment_time_range(0.7, 1.3)
    ss.sync_flag(False)
    
    # build model
    T = ss.get_frames()
    F = n_mels
    C = ss.get_number_of_channels()
    D = 20
    model = chimeranet.model.build_model(T, F, C, D)
    model.compile(
        'rmsprop',
        loss={
            'embedding': chimeranet.model.loss_deepclustering,
            'mask': chimeranet.model.loss_mask
        },
        loss_weights={
            'embedding': 0.5,
            'mask': 0.5
        }
    )

    # obtain training data from spectrogram sampler and train model
    sample_size, batch_size = 256, 1
    specs = ss.make_specs(sample_size)
    with h5py.File(dataset_path, 'w') as f:
        f.create_dataset('specs', data=specs)
    x = to_mixture(specs)
    y = to_true_pair(specs)
    model.fit(x=x, y=y, batch_size=batch_size, epochs=200)
    model.save(model_path)

if __name__ == '__main__':
    main()
