
import os
import sys
import random
import numpy as np
import librosa
import matplotlib.pyplot as plt
from itertools import permutations
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Test ChimeraNet model')
    parser.add_argument('--voc-dir', metavar='path', help='directory contains voice wav files', required=True)
    parser.add_argument('--mel-dir', metavar='path', help='directory contains melody wav files', required=True)
    parser.add_argument('--out-dir', metavar='path', help='directory will contain result', required=True)
    parser.add_argument('--model', metavar='path', help='input model file', required=True)
    parser.add_argument('--duration', metavar='T', help='time per model sample (s)', type=float, default=0.5)
    parser.add_argument('--test-duration', metavar='T', help='time per test sample (s)', type=float, default=1)
    parser.add_argument('--embd-dim', metavar='D', help='dimension of embed vector', type=int, default=20)
    parser.add_argument('--n-mels', metavar='n', help='number of melspectrogram bins', type=int, default=64)
    parser.add_argument('-q', '--quiet', help='quiet mode (only supported on TF backend)', action='store_true')
    advanced = parser.add_argument_group('advanced')
    advanced.add_argument('--sr', metavar='sr', help='sampling rate', type=int, default=16000)
    advanced.add_argument('--hop-length', metavar='n', help='hop length', type=int, default=128)
    advanced.add_argument('--n-fft', metavar='n', help='', type=int, default=512)
    return parser.parse_args()

stdout, stderr = sys.stdout, sys.stderr
def quiet():
    sys.stdout, sys.stderr = open(os.devnull, 'w'), open(os.devnull, 'w')
def verbose():
    sys.stdout, sys.stderr = stdout, stderr

args = parse_args()
if args.quiet:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    quiet()

from keras.models import load_model
from keras.utils import CustomObjectScope
from feature import from_feature
from model import make_loss_dc, make_loss_mmsa, build_model
from data_generator import generate_test_data
from data_util import split_window, find_embd_matrix, find_mask

def save(v, m, mix, mixp, mel_basis_inv, dir, prefix, sr, C=2):
    if not os.path.isdir(dir):
        os.makedirs(dir)
    # reconstruct from mask
    mask = tuple(map(lambda x: x[:, :mix.shape[1]], find_mask(m)))
    for c, m_c in zip(('voc', 'mel'), mask):
        wav_path = os.path.join(dir, '{}-mask-{}.wav'.format(prefix, c))
        y = from_feature(m_c*mix, mixp, mel_basis_inv)
        librosa.output.write_wav(wav_path, y, sr)
        fig_path = os.path.join(dir, '{}-mask-{}.png'.format(prefix, c))
        plt.subplot(2, 1, 1)
        plt.imshow(m_c, vmin=0, vmax=1)
        plt.gca().invert_yaxis()
        plt.subplot(2, 1, 2)
        plt.imshow(m_c*mix, vmin=0, vmax=1)
        plt.gca().invert_yaxis()
        plt.savefig(fig_path)
        plt.close()

    # reconstruct from embedding
    embd = np.array(
        tuple(map(lambda x: x[:, :mix.shape[1]],
                  find_embd_matrix(v, C)))
    )
    perm = min(
        permutations(range(C)),
        key=lambda p: np.sum((embd.take(p, axis=0)-mask)**2)
    )
    for c, v_c in zip(('voc', 'mel'), embd.take(perm, axis=0)):
        wav_path = os.path.join(dir, '{}-embd-{}.wav'.format(prefix, c))
        y = from_feature(v_c*mix, mixp, mel_basis_inv)
        librosa.output.write_wav(wav_path, y, sr)
        fig_path = os.path.join(dir, '{}-embd-{}.png'.format(prefix, c))
        plt.subplot(2, 1, 1)
        plt.imshow(v_c, vmin=0, vmax=1)
        plt.gca().invert_yaxis()
        plt.subplot(2, 1, 2)
        plt.imshow(v_c*mix, vmin=0, vmax=1)
        plt.gca().invert_yaxis()
        plt.savefig(fig_path)
        plt.close()

def main():
    T = librosa.core.time_to_frames(args.duration, args.sr, args.hop_length) + 1
    F = args.n_mels
    C = 2
    D = args.embd_dim

    mel_basis = librosa.filters.mel(args.sr, args.n_fft, args.n_mels)
    mel_basis_inv = np.linalg.pinv(mel_basis)

    raws, targets = next(generate_test_data(
        args.voc_dir, args.mel_dir, n_mels=args.n_mels, duration=args.test_duration
    ))
    raws = tuple(map(lambda X: (X[0][0], X[1][0]), raws))
    mix, mixp = raws[2]

    with CustomObjectScope({
        'loss_dc': make_loss_dc(F, T, C, D),
        'loss_mmsa': make_loss_mmsa(F, T, C, D)
    }):
        model = load_model(args.model)
    vhat, mhat = model.predict(split_window(mix, T))

    # write true
    save(
        np.transpose(targets['V'], (0, 3, 2, 1)),
        np.transpose(targets['M'][:, :C, :, :], (0, 3, 2, 1)),
        mix, mixp, mel_basis_inv, args.out_dir, 'true', args.sr
    )
    # write predicted
    save(vhat, mhat, mix, mixp, mel_basis_inv, args.out_dir, 'pred', args.sr)

if __name__ == '__main__':
    main()
    pass
