#!/usr/bin/env python3

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-c', '--channel', action='append', nargs='+', required=True,
        metavar='INFO',
        help='List of channel info: NAME TYPE PATH [ARCHIVE_PATH]'
    )
    parser.add_argument(
        '-o', '--output', type=str, required=True,
        metavar='PATH',
        help='Output path'
    )
    parser.add_argument(
        '-t', '--time', type=float, default=.5,
        metavar='T',
        help='Window length in second (default=.5)'
    )
    parser.add_argument(
        '-f', '--freq-bins', type=int, default=64,
        metavar='F',
        help='Number of mel-spectrogram bins (default=64)'
    )
    parser.add_argument(
        '--sr', type=int, default=16000,
        metavar='N',
        help='Sampling rate (default=16000)'
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

    sync_group = parser.add_mutually_exclusive_group()
    sync_group.add_argument(
        '--sync', action='store_true', default=True,
        help='Synchronized dataset'
    )
    sync_group.add_argument(
        '--async', dest='sync', action='store_false',
        help='Asynchronized dataset'
    )
    concrete_group = parser.add_mutually_exclusive_group()
    concrete_group.add_argument(
        '--concrete', action='store_true', default=True,
        help='Generate dataset on this program'
    )
    concrete_group.add_argument(
        '--phony', dest='concrete', action='store_false',
        help='Generate training data on training phase'
    )

    augmentation_group = parser.add_argument_group(title='data augmentation')
    augmentation_group.add_argument(
        '--augment-time', nargs=2, type=float, default=(1., 1.),
        metavar='VAL',
        help='Augment range in time space. If 0.5 2 are given, '
    )
    augmentation_group.add_argument(
        '--augment-freq', nargs=2, type=float, default=(0., 0.),
        metavar='VAL',
        help='Augment range in frequency space. If 0.5 2 are given, '
    )
    augmentation_group.add_argument(
        '--augment-amplitude', nargs=2, type=float, default=(0., 0.),
        metavar='VAL',
        help='Augment range of amplitude. If 0.5 2 are given, '
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)

if __name__ == '__main__':
    main()
