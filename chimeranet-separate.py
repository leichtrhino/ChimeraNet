
import os
import sys
import random
import numpy as np
import librosa
from itertools import permutations
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Separator using ChimeraNet')
    parser.add_argument('-i', '--input', metavar='path', help='input files', nargs='*')
    parser.add_argument('-o', '--output', metavar='path', help='output files', nargs='*')
    parser.add_argument('-m', '--model-path', metavar='path', help='input model file', required=True)
    parser.add_argument('--duration', metavar='T', help='time per model sample (s)', type=float, default=0.5)
    parser.add_argument('--embd-dim', metavar='D', help='dimension of embed vector', type=int, default=20)
    parser.add_argument('--n-mels', metavar='n', help='number of melspectrogram bins', type=int, default=64)
    parser.add_argument('-q', '--quiet', help='quiet mode (only supported on TF backend)', action='store_true')
    parser.add_argument('--with-embd', action='store_true')
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
from feature import from_feature, to_feature
from model import make_loss_dc, make_loss_mmsa, build_model
from data_generator import generate_test_data
from data_util import split_window, find_embd_matrix, find_mask

# parameters
T = librosa.core.time_to_frames(args.duration, args.sr, args.hop_length) + 1
F = args.n_mels
C = 2
D = args.embd_dim

mel_basis = librosa.filters.mel(args.sr, args.n_fft, args.n_mels)
mel_basis_inv = np.linalg.pinv(mel_basis)

with CustomObjectScope({
    'loss_dc': make_loss_dc(F, T, C, D),
    'loss_mmsa': make_loss_mmsa(F, T, C, D)
}):
    print('Loading model "{}"...'.format(args.model_path))
    try:
        model = load_model(args.model_path)
    except Exception as e:
        verbose()
        raise e

if args.output:
    if len(args.input) > 1 and len(args.output) == 1:
        opdir = args.output[0]
        assert(not os.path.exists(opdir) or os.path.isdir(opdir))
        if not os.path.exists(opdir):
            os.makedirs(opdir)
    else:
        assert(len(args.input) == len(args.output))
        ops = iter(args.output)
formated_path = lambda a, b, c:\
                '{}-{}-{}.wav'.format(a, b, c) if args.with_embd else\
                '{}-{}.wav'.format(a, b)

for ip in args.input:
    if args.output:
        if len(args.input) > 1 and len(args.output) == 1:
            op = os.path.join(opdir, os.path.splitext(os.path.basename(ip))[0])
        else:
            op = os.path.splitext(next(ops))[0]
    else:
        op = os.path.splitext(ip)[0]
    print('Processing "{}" -> "{}"...'.format(ip, op))
    
    mix, mixp = to_feature(librosa.load(ip, sr=args.sr)[0], mel_basis, True)
    vhat, mhat = model.predict(split_window(mix, T))
    
    # reconstruct from mask
    mask = tuple(map(lambda x: x[:, :mix.shape[1]], find_mask(mhat)))
    for c, mhat_c in zip(('voc', 'mel'), mask):
        y = from_feature(mhat_c*mix, mixp, mel_basis_inv)
        librosa.output.write_wav(formated_path(op, c, 'mask'), y, args.sr)
    
    # reconstruct from embedding
    if args.with_embd:
        embd = np.array(
            tuple(map(lambda x: x[:, :mix.shape[1]],
                      find_embd_matrix(vhat, C)))
        )
        perm = min(
            permutations(range(C)),
            key=lambda p: np.sum((embd.take(p, axis=0)-mask)**2)
        )
        for c, vhat_c in zip(('voc', 'mel'), embd.take(perm, axis=0)):
            y = from_feature(vhat_c*mix, mixp, mel_basis_inv)
            librosa.output.write_wav(formated_path(op, c, 'embd'), y, args.sr)
