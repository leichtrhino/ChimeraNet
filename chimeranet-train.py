
import os
import sys
import random
import itertools
import pickle
import numpy as np
import librosa
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Train ChimeraNet model')
    parser.add_argument('--voc-dir', metavar='path', help='directory contains voice wav files', required=True)
    parser.add_argument('--mel-dir', metavar='path', help='directory contains melody wav files', required=True)
    parser.add_argument('--model', metavar='path', help='output model file', required=True)
    parser.add_argument('--duration', metavar='T', help='time per a sample (s)', type=float, default=0.5)
    parser.add_argument('--embd-dim', metavar='D', help='dimension of embed vector', type=int, default=20)
    parser.add_argument('--n-mels', metavar='n', help='number of melspectrogram bins', type=int, default=64)
    parser.add_argument('--batch-size', metavar='n', type=int, default=8)
    parser.add_argument('--steps', metavar='n', type=int, default=7200//8)
    parser.add_argument('--epochs', metavar='n', type=int, default=1)
    parser.add_argument('-q', '--quiet', help='quiet mode (only supported on TF backend)', action='store_true')
    advanced = parser.add_argument_group('advanced')
    advanced.add_argument('--sr', metavar='sr', help='sampling rate', type=int, default=16000)
    advanced.add_argument('--hop-length', metavar='n', help='hop length', type=int, default=128)
    advanced.add_argument('--checkpoint-format', metavar='format', help='format for ModelCheckPoint (e.g. \'model.{epoch:04d}.hdf5)\'')
    advanced.add_argument('--history', metavar='path', help='output history path')
    advanced.add_argument('--validation-steps', metavar='n', type=int)
    advanced.add_argument('--validation-voc-dir', metavar='path')
    advanced.add_argument('--validation-mel-dir', metavar='path')
    advanced.add_argument('--previous-model', metavar='path', help='pretrained model')
    advanced.add_argument('--previous-epoch', metavar='n', help='epochs of pretrained model', type=int)
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

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import CustomObjectScope

from feature import to_feature
from model import make_loss_dc, make_loss_mmsa, build_model
from data_generator import generate_data

def main():
    F = args.n_mels
    T = librosa.core.time_to_frames(args.duration, args.sr, args.hop_length) + 1
    D = args.embd_dim
    C = 2

    if not args.previous_model:
        model = build_model(F, T, C, D)
        model.compile(
            'rmsprop',
            loss={'V': make_loss_dc(F, T, C, D), 'M': make_loss_mmsa(F, T, C, D)},
            loss_weights={'V': 0.5, 'M': 0.5}
        )
    else:
        with CustomObjectScope({
            'loss_dc': make_loss_dc(F, T, C, D), 'loss_mmsa': make_loss_mmsa(F, T, C, D)
        }):
            model = load_model(args.previous_model)

    train_generator = generate_data(args.voc_dir, args.mel_dir, k=args.batch_size, duration=args.duration, n_mels=args.n_mels)
    if all((args.validation_voc_dir, args.validation_mel_dir, args.validation_steps)):
        validation_generator = generate_data(args.validation_voc_dir, args.validation_mel_dir, k=args.batch_size, duration=args.duration, n_mels=args.n_mels)
    else:
        validation_generator = None
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=args.steps,
        validation_data=validation_generator,
        validation_steps=args.validation_steps if validation_generator else None,
        epochs=args.epochs,
        initial_epoch=args.previous_epoch if args.previous_model else 0,
        callbacks=[ModelCheckpoint(args.checkpoint_format)] if args.checkpoint_format else None
    )
    model.save(args.model)
    if args.history:
        with open(args.history, 'wb') as fp:
            pickle.dump(history.history, fp)

if __name__ == '__main__':
    main()
    
