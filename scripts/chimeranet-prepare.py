#!/usr/bin/env python3

import re
import os
import sys
import importlib
import h5py
import librosa
from argparse import ArgumentParser, SUPPRESS

def main():
    args = parse_args()

    if not importlib.util.find_spec('chimeranet'):
        print('ChimeraNet is not installed, import from source.')
        sys.path.append(os.path.join(os.path.split(__file__)[0], '..'))

    from chimeranet import Sampler, DatasetSampler, AsyncSampler, SyncSampler
    from chimeranet import to_training_data

    datasets = [to_dataset(c) for c in args.channels]
    samplerclass = SyncSampler if args.sync else AsyncSampler
    sampler = samplerclass(*datasets)
    sampler.duration = args.time
    sampler.samplerate = args.sr
    samples = sampler.sample(args.sample_size, n_jobs=-1)

    T = librosa.time_to_frames(
        args.time, args.sr, args.hop_length, args.n_fft)
    F = args.freq_bins
    x, y = to_training_data(
        samples, T, F, n_jobs=-1,
        sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length,
    )

    with h5py.File(args.output, 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('y/mask', data=y['mask'])
        f.create_dataset('y/embedding', data=y['embedding'])
        if args.save_audio:
            f.create_dataset('audio', data=samples)

def to_dataset(channel_info):
    from chimeranet import DirDataset, ZipDataset, TarDataset
    from chimeranet import VoxCeleb, DSD100Melody, DSD100Vocal, ESC50, LMD
    
    def to_dataset_part(path, tag):
        if not os.path.exists(path):
            raise RuntimeError('"{}" not found'.format(path))
        base, ext = os.path.splitext(os.path.basename(path).lower())
        if ext and ext not in ('.zip', '.gz'):
            raise RuntimeError('"{}" is not supported extension'.format(path))
        if not ext and not os.path.isdir(path):
            raise RuntimeError('"{}" is not a directory'.format(path))
        
        is_val = any(
            t.lower().startswith('val') or t.lower().startswith('test')
            for t in tag
        )
        if 'dsd100' in base:
            is_voc = any(t.lower().startswith('voc') for t in tag)
            is_mel = any(t.lower().startswith('mel') for t in tag)
            if is_voc and is_mel:
                raise RuntimeError('DSD100 needs mel or voc as tag')
            elif not is_voc and not is_mel:
                raise RuntimeError('DSD100 needs mel or voc as tag')
            dclass = DSD100Vocal if is_voc else DSD100Melody
            return dclass(path, dev=not is_val, test=is_val)
        elif 'esc-50' in base or 'esc50' in base:
            return ESC50(
                path, ESC50.all_categories(),
                fold=(5,) if is_val else (1, 2, 3, 4)
            )
        elif 'voxceleb' in base:
            return VoxCeleb(path)
        elif ext == '.zip':
            return ZipDataset(path)
        elif ext == '.gz':
            return TarDataset(path)
        else:
            return DirDataset(path)
    datasets = [
        to_dataset_part(path, channel_info.get('tag', []))
        for path in channel_info['path']
    ]
    return sum(datasets[1:], datasets[0])

def parse_args():
    n_channels = 4
    parser = ArgumentParser()
    for ci in range(1, n_channels+1):
        if ci <= 2:
            help = 'Channel {} information. PATH and TAG are csv.'.format(ci)
        else:
            help = SUPPRESS
        parser.add_argument(
            '--c{}'.format(ci),
            metavar='PATH [TAG]',
            help=help
        )
    parser.add_argument(
        '-o', '--output', type=str, required=True,
        metavar='PATH',
        help='Output path'
    )
    parser.add_argument(
        '-n', '--sample-size', type=int, required=True,
        metavar='N',
        help='Sample size'
    )
    parser.add_argument(
        '-t', '--time', type=float, default=2.,
        metavar='T',
        help='Window length in second (default=2.)'
    )
    parser.add_argument(
        '-f', '--freq-bins', type=int, default=150,
        metavar='F',
        help='Number of mel-spectrogram bins (default=150)'
    )
    parser.add_argument(
        '--sr', type=int, default=44100,
        metavar='N',
        help='Sampling rate (default=44100)'
    )
    parser.add_argument(
        '--n-fft', type=int, default=2048,
        metavar='F',
        help='FFT window size (default=2048)'
    )
    parser.add_argument(
        '--hop-length', type=int, default=512,
        metavar='N',
        help='Hop length on STFT (default=512)'
    )
    parser.add_argument(
        '--save-audio', action='store_true',
        help='Save audio to "audio" dataset'
    )

    sync_group = parser.add_mutually_exclusive_group()
    sync_group.add_argument(
        '--sync', action='store_true', default=True,
        help='Synchronized dataset'
    )
    sync_group.add_argument(
        '--async', dest='sync', action='store_false',
        help='Asynchronized dataset'
    )

    args = parser.parse_args()
    args.channels = list(filter(
        lambda x: x,
        map(
            lambda ci: getattr(args, 'c{}'.format(ci)),
            range(1, n_channels+1)
        )
    ))
    for ci in range(1, n_channels+1):
        delattr(args, 'c{}'.format(ci))
    if len(args.channels) < 2:
        parser.error('2 or more channel information are required')
    for i, c in enumerate(args.channels):
        c = [b.strip() for b in re.split('[,;+]', c)]
        j = -1
        try:
            j = next(k for k, b in enumerate(c) if ' ' in b)
        except StopIteration:
            pass
        if j >= 0:
            d = {
                'path': c[:j]+[c[j].split()[0].strip()],
                'tag': [c[j].split()[1].strip()]+c[j+1:]
            }
        else:
            d = {'path': c}
        args.channels[i] = d
    return args

if __name__ == '__main__':
    main()
