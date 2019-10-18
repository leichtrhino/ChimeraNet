#!/usr/bin/env python3
import os
import sys
import glob
import importlib
import numpy as np
import librosa

from argparse import ArgumentParser
from itertools import permutations

def main():
    args = parse_args()
    input_paths = sum((glob.glob(f) for f in args.input_audio), [])

    if not importlib.util.find_spec('chimeranet'):
        print('ChimeraNet is not installed, import from source.')
        sys.path.append(os.path.join(os.path.split(__file__)[0], '..'))
    from chimeranet.models import probe_model_shape, load_model

    args.n_frames, args.n_mels, args.n_channels, args.d_embedding\
        = probe_model_shape(args.model_path)
    if len(args.channel_name) < args.n_channels:
        raise ValueError # short channel names.
    args.model = load_model(args.model_path)
    if 0 < args.n_mels < args.n_fft // 2 + 1:
        args.mel_basis = librosa.filters.mel(
            args.sr, args.n_fft, args.n_mels, norm=None
        )

    for input_path in input_paths:
        part(input_path, **args.__dict__)

def part(input_path, **kwargs):
    print('processing {}'.format(input_path))

    from chimeranet import from_embedding, from_mask
    from chimeranet import split_window
    from chimeranet import merge_windows_mean, merge_windows_most_common
    if kwargs['plot_spectrograms']:
        import matplotlib.pyplot as plt

    # load audio and split into windows
    audio, _ = librosa.core.load(
        input_path, sr=kwargs['sr'], duration=kwargs['duration'])
    spec, phase = librosa.core.magphase(
        librosa.core.stft(audio, kwargs['n_fft'], kwargs['hop_length'])
    )
    if 0 < kwargs['n_mels'] < kwargs['n_fft'] // 2 + 1:
        spec = np.dot(kwargs['mel_basis'], spec)

    mask_embd = np.empty((kwargs['n_channels'], kwargs['n_mels'], 0))
    mask_mask = np.empty((kwargs['n_channels'], kwargs['n_mels'], 0))
    n_batch \
        = int(np.ceil(
            spec.shape[1] / kwargs['n_frames'] / kwargs['batch_size']
        )) if kwargs['batch_size'] > 0 else 0
    for batch_i in range(max(n_batch, 1)):
        if n_batch == 0:
            s = spec
        else:
            window_size = kwargs['batch_size']*kwargs['n_frames']
            s = spec[:, batch_i*window_size:(batch_i+1)*window_size]
        if s.shape[1] < kwargs['n_frames']:
            s = np.hstack(
                (s, np.zeros((s.shape[0], kwargs['n_frames']-s.shape[1])))
            )
        x = split_window(s, kwargs['n_frames']).transpose((0, 2, 1))

        # predict
        embedding, mask = kwargs['model'].predict(x)
        y_embd = from_embedding(
            embedding, kwargs['n_channels'], n_jobs=kwargs['jobs'])
        y_mask = from_mask(mask)
        mini_mask_embd = merge_windows_most_common(y_embd)[:, :, :s.shape[1]]
        mini_mask_mask = merge_windows_mean(y_mask)[:, :, :s.shape[1]]
        ordered_mini_mask_embd = min(
            (t for t in permutations(mini_mask_embd)),
            key=lambda t: np.sum(np.abs(np.array(t)-mini_mask_mask)*s)
        )
        mask_embd = np.dstack((mask_embd, ordered_mini_mask_embd))
        mask_mask = np.dstack((mask_mask, mini_mask_mask))
    mask_embd = mask_embd[:, :, :spec.shape[1]]
    mask_mask = mask_mask[:, :, :spec.shape[1]]

    # reconstruct from prediction
    if kwargs['plot_spectrograms']:
        fig = plt.figure(figsize=(30, 15))
        nrow = 2*kwargs['n_inference']+1
        ncol = kwargs['n_channels']
        ax = fig.add_subplot(nrow, ncol, 1)
        ax.title.set_text('spec. of '+input_path)
        ax.imshow(spec, origin='lower', aspect='auto')

    i_inference = 0
    if not kwargs['disable_mask_output']:
        for ci, mask in enumerate(mask_mask):
            save_audio(
                input_path, mask, spec, phase,
                i_inference=i_inference, i_channel=ci, **kwargs
            )
            if kwargs['plot_spectrograms']:
                plot_spec(
                    input_path, mask, spec, fig,
                    i_inference=i_inference, i_channel=ci, **kwargs
                )                
        i_inference += 1

    if not kwargs['disable_embedding_inference']:
        # to align embedding to mask
        mask_embd_ordered = min(
            (t for t in permutations(mask_embd)),
            key=lambda t: np.sum(np.abs(np.array(t)-mask_mask)*spec)
        )
        for ci, mask in enumerate(mask_embd_ordered):
            save_audio(
                input_path, mask, spec, phase,
                i_inference=i_inference, i_channel=ci, **kwargs
            )
            if kwargs['plot_spectrograms']:
                plot_spec(
                    input_path, mask, spec, fig,
                    i_inference=i_inference, i_channel=ci, **kwargs
                )                
        i_inference += 1

    if kwargs['plot_spectrograms']:
        output_plot_path = kwargs['output_plot_mapper'](input_path)
        output_plot_dir = os.path.dirname(output_plot_path)
        if not os.path.exists(output_plot_dir):
            os.makedirs(output_plot_dir)
        elif not os.path.isdir(output_plot_dir):
            print('warning: {} will not be created.'.format(output_plot_dir)) # TODO
        if os.path.isdir(output_plot_dir):
            plt.savefig(output_plot_path)

def plot_spec(input_path, mask, spec, fig, **kwargs):
    pred_spec = mask * spec
    output_audio_path = kwargs['output_audio_mapper'](
        input_path, kwargs['i_inference'], kwargs['i_channel']
    )
    nrow = 2*kwargs['n_inference']+1
    ncol = kwargs['n_channels']

    ax = fig.add_subplot(
        nrow, ncol,
        (2*kwargs['i_inference']+1)*ncol+kwargs['i_channel']+1
    )
    ax.title.set_text('mask of ' + output_audio_path)
    ax.imshow(mask, origin='lower', aspect='auto')

    ax = fig.add_subplot(
        nrow, ncol,
        (2*kwargs['i_inference']+2)*ncol+kwargs['i_channel']+1
    )
    ax.title.set_text('spec. of ' + output_audio_path)
    ax.imshow(pred_spec, origin='lower', aspect='auto')

def save_audio(input_path, mask, spec, phase, **kwargs):
    pred_spec = mask * spec
    if 0 < kwargs['n_mels'] < kwargs['n_fft'] // 2 + 1:
        pred_spec = np.dot(kwargs['mel_basis'].T, pred_spec)        
    output_audio_path = kwargs['output_audio_mapper'](
        input_path, kwargs['i_inference'], kwargs['i_channel']
    )
    output_audio_dir = os.path.dirname(output_audio_path)
    if not os.path.exists(output_audio_dir):
        os.makedirs(output_audio_dir)
    elif not os.path.isdir(output_audio_dir):
        print('warning: {} will not be created.'.format(output_audio_dir)) # TODO
    if os.path.isdir(output_audio_dir):
        out_audio = librosa.core.istft(pred_spec*phase, kwargs['hop_length'])
        librosa.output.write_wav(output_audio_path, out_audio, kwargs['sr'])

def parse_args():
    parser = ArgumentParser()
    # basic arguments
    basic_group = parser.add_argument_group(title='basic arguments')
    basic_group.add_argument(
        '-m', '--model-path', type=str, metavar='PATH',
        required=True, 
    )
    basic_group.add_argument(
        '-i', '--input-audio', type=str, nargs='*', metavar='PATH',
        required=True,
    )
    basic_group.add_argument(
        '-o', '--output-audio', type=str, metavar='FORMATTED STRING',
        default='{input:}_{infer:}_{channel:}.wav',
        help='''Formatted string as output audio path
(e.g. "{input:}_{infer:}_{channel:}.wav" (default)).'''
    )
    basic_group.add_argument(
        '-d', '--output-directory', type=str, metavar='DIR',
        help='If specified, add it as top directory of "--output-audio"'
    )
    basic_group.add_argument(
        '--batch-size', type=int, metavar='N', default=0,
        help='Batch size on separation'
    )

    # audio arguments
    audio_group = parser.add_argument_group(title='audio arguments')
    audio_group.add_argument(
        '--sr', type=int, default=16000,
        metavar='N',
        help='Sampling rate (default=16000)'
    )
    audio_group.add_argument(
        '--n-fft', type=int, default=512,
        metavar='F',
        help='FFT window size (default=512)'
    )
    audio_group.add_argument(
        '--hop-length', type=int, default=128,
        metavar='N',
        help='Hop length on STFT (default=128)'
    )
    audio_group.add_argument(
        '--duration', type=float, default=0.,
        metavar='T',
        help='Audio duration in seconds'
    )

    # advanced output arguments
    advanced_output_group = parser.add_argument_group(title='advanced output')
    advanced_output_group.add_argument(
        '--replace-top-directory', type=str, metavar='DIR',
        help='If specified, replace top directory of "--output-audio" with it'
    )
    advanced_output_group.add_argument(
        '--plot-spectrograms', action='store_true',
        help='Enable output spectrograms'
    )
    advanced_output_group.add_argument(
        '--disable-embedding-inference', action='store_true',
        help='Disable embedding inference'
    )
    advanced_output_group.add_argument(
        '--disable-mask-output', action='store_true',
        help='Disable output from mask inference'
    )
    advanced_output_group.add_argument(
        '--output-plot', type=str, metavar='FORMATTED STRING',
        default='{input:}.png',
        help='''Formatted string as output spectrogram plot prefix
(e.g. "{input:}.png").'''
    )
    advanced_output_group.add_argument(
        '--channel-name', type=str, nargs='*', metavar='NAME',
        help='Channel names show on output audio and/or plot.'
    )
    advanced_output_group.add_argument(
        '--embedding-inference-name', type=str, metavar='NAME',
        default='embd',
        help='Inference name of embedding',
    )
    advanced_output_group.add_argument(
        '--mask-inference-name', type=str, metavar='NAME',
        default='mask',
        help='Inference name of embedding',
    )
    advanced_output_group.add_argument(
        '-j', '--jobs', type=int, metavar='N',
        default=1,
        help='The number of jobs of k-means clustering',
    )

    args = parser.parse_args()
    if not args.duration:
        args.duration = None
    args.inference_name = [
        args.mask_inference_name,
        args.embedding_inference_name,
    ]
    if args.disable_embedding_inference:
        args.inference_names.pop(1)
    if args.disable_mask_output:
        args.inference_names.pop(0)
    args.n_inference = len(args.inference_name)
    if args.channel_name is None:
        class ChannelNameMapper(object):
            def __getitem__(self, key):
                return 'ch{key:}'.format(key=key+1)
            def __len__(self):
                return 10000 # sufficiently long list
        args.channel_name = ChannelNameMapper()

    try:
        args.output_audio.format(
            input=os.path.splitext(args.input_audio[0])[0],
            infer=args.mask_inference_name,
            channel=args.channel_name[0],
        )
    except KeyError:
        parser.error(
            '"--output-audio" must not take other than '
            '"input", "infer" and "channel" as key.'
        )
    if args.output_directory and args.replace_top_directory:
        parser.error(
            '"--output-directory" and "--replace-top-directory" are '
            'mutually exclusive.'
        )
    def output_audio_mapper(input, i_infer, i_channel):
        output_path = args.output_audio.format(
            input=os.path.splitext(input)[0],
            infer=args.inference_name[i_infer],
            channel=args.channel_name[i_channel]
        )
        if args.output_directory:
            output_path = os.path.join(args.output_directory, output_path)
        if args.replace_top_directory:
            output_path = os.path.join(
                args.replace_top_directory,
                *output_path.split(os.path.sep)[1:]
            )
        return output_path
    args.output_audio_mapper = output_audio_mapper

    if args.plot_spectrograms:
        if not importlib.util.find_spec('matplotlib'):
            parser.error(
                '"--plot-spectrogram": matplotlib is not installed.'
            )
        try:
            args.output_plot.format(
                input=os.path.splitext(args.input_audio[0])[0],
            )
        except KeyError:
            parser.error(
                '"--output-plot" must not take other than "input" as key.'
            )
        def output_plot_mapper(input):
            output_path = args.output_plot.format(
                input=os.path.splitext(input)[0]
            )
            if args.output_directory:
                output_path = os.path.join(args.output_directory, output_path)
            if args.replace_top_directory:
                output_path = os.path.join(
                    args.replace_top_directory,
                    *output_path.split(os.path.sep)[1:]
                )
            return output_path
        args.output_plot_mapper = output_plot_mapper

    return args

if __name__ == '__main__':
    main()
