#!/usr/bin/env python3

import os
import sys
import h5py
import numpy as np
import librosa
import multiprocessing
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def main():
    args = parse_args()

    with h5py.File(args.input, 'r') as f:
        x = f['x'][()]
        y_embedding = f['y/embedding'][()]
        y_mask = f['y/mask'][()]
        if 'audio' in f.keys():
            audio = f['audio'][()]
        else:
            audio = [None] * x.shape[0]
    
    x = x.transpose((0, 2, 1))
    y_embedding = y_embedding.transpose((0, 3, 2, 1))
    y_mask = y_mask.transpose((0, 3, 2, 1))

    args_list = [
        (index, x_, y_e_, y_m_, a, args) for index, (x_, y_e_, y_m_, a)
        in enumerate(zip(x, y_embedding, y_mask, audio))
    ]
    list(map(process_part, args_list))

def process_part(args):
    index, x, y_embedding, y_mask, audio, args = args

    fig = plt.figure(figsize=(32, 16))
    n_channel = y_embedding.shape[0]
    ax = fig.add_subplot(3, n_channel, 1)
    ax.title.set_text('mixture')
    im = ax.imshow(x, origin='lower', aspect='auto')
    plt.colorbar(im)

    for ri, (name, y) in enumerate(zip(
        ('deepclustering', 'mask'), (y_embedding, y_mask[:n_channel])
    ), 1):
        for ci, m in enumerate(y, 1):
            ax = fig.add_subplot(3, n_channel, ri*n_channel+ci)
            ax.title.set_text('{} of {}'.format(name, args.channel_name[ci]))
            ax.imshow(m, origin='lower', aspect='auto', vmin=0, vmax=1)

    path = args.output_plot_mapper(index)
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    fig.savefig(path)
    plt.close(fig)

    if args.write_audio and audio is not None:
        for ci, a in enumerate(audio):
            path = args.output_audio_mapper(index, ci)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            librosa.output.write_wav(path, a, args.sr)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=str, metavar='PATH', required=True,
    )
    parser.add_argument(
        '--output-plot', type=str, metavar='FORMATTED STRING',
        default='{index:}.png',
        help='''Formatted string as output spectrogram plot prefix
(e.g. "{index:}.png").'''
    )
    parser.add_argument(
        '--write-audio', action='store_true',
        help='Enable output audio if exists'
    )
    parser.add_argument(
        '--output-audio', type=str, metavar='FORMATTED STRING',
        default='{index:}_{channel:}.wav',
        help='''Formatted string as output audio path
(e.g. "{index:}_{channel:}.wav" (default)).'''
    )
    parser.add_argument(
        '-d', '--output-directory', type=str, metavar='DIR',
        help='If specified, add it as top directory of "--output-audio"'
    )
    parser.add_argument(
        '--sr', type=int, default=44100,
        metavar='N',
        help='Sampling rate (default=44100)'
    )
    parser.add_argument(
        '--channel-name', type=str, nargs='*', metavar='NAME',
        help='Channel names show on output audio and/or plot.'
    )

    args = parser.parse_args()
    if args.channel_name is None:
        class ChannelNameMapper(object):
            def __getitem__(self, key):
                return 'ch{key:}'.format(key=key+1)
            def __len__(self):
                return 10000 # sufficiently long list
        args.channel_name = ChannelNameMapper()

    try:
        args.output_plot.format(
            index=0,
        )
    except KeyError:
        parser.error(
            '"--output-plot" must not take other than "index" as key.'
        )
    def output_plot_mapper(index):
        output_path = args.output_plot.format(
            index=index
        )
        if args.output_directory:
            output_path = os.path.join(args.output_directory, output_path)
        return output_path
    args.output_plot_mapper = output_plot_mapper

    if args.write_audio:
        try:
            args.output_audio.format(
                index=0,
                channel=args.channel_name[0],
            )
        except KeyError:
            parser.error(
                '"--output-audio" must not take other than '
                '"input" and "channel" as key.'
            )
        def output_audio_mapper(index, i_channel):
            output_path = args.output_audio.format(
                index=index,
                channel=args.channel_name[i_channel]
            )
            if args.output_directory:
                output_path = os.path.join(args.output_directory, output_path)
            return output_path
        args.output_audio_mapper = output_audio_mapper
    return args

if __name__ == '__main__':
    main()
