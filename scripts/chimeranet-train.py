#!/usr/bin/env python3

import os
import sys
import importlib
import h5py
import random
import numpy as np
from argparse import ArgumentParser

def main():
    args = parse_args()

    if not importlib.util.find_spec('chimeranet'):
        print('ChimeraNet is not installed, import from source.')
        sys.path.append(os.path.join(os.path.split(__file__)[0], '..'))

    from keras.callbacks import Callback, CSVLogger
    from chimeranet.models import probe_model_shape, load_model, ChimeraPPModel
    class NameDataset(Callback):
        def __init__(self, name, val_name=None):
            self.name = name
            self.val_name = val_name
        def on_epoch_end(self, epoch, logs):
            logs['dataset'] = self.name
            logs['val_dataset'] = self.val_name

    with h5py.File(args.train_data, 'r') as f:
        _, T, F, C = f['y/embedding'].shape

    # build/load model
    if args.input_model is not None:
        T_, F_, C_, D_ = probe_model_shape(args.input_model)
        assert T == T_ and F == F_ and C == C_,\
            'Incompatible dataset with the model'
        model = load_model(args.input_model)
    else:
        cm = ChimeraPPModel(T, F, C, args.embedding_dims)
        model = cm.build_model()
        model.compile(
            'rmsprop',
            loss={
                'embedding': cm.loss_deepclustering(),
                'mask': cm.loss_mask()
            },
            loss_weights={
                'embedding': 0.9,
                'mask': 0.1
            }
        )

    # train
    train_generator = generate_data(
        args.train_data, args.batch_size, shuffle=True)
    train_steps = get_dataset_size(
        args.train_data) // args.batch_size

    if args.validation_data:
        validation_generator = generate_data(
            args.validation_data, args.batch_size)
        validation_steps = get_dataset_size(
            args.validation_data) // args.batch_size
    else:
        validation_generator = None
        validation_steps = None

    model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        initial_epoch=args.initial_epoch,
        epochs=args.stop_epoch,
        callbacks=[
            NameDataset(args.train_data, args.validation_data),
            CSVLogger(args.log, append=args.initial_epoch > 0),
        ],
    )
    model.save(args.output_model)

def get_dataset_size(filename):
    with h5py.File(filename, 'r') as f:
        sample_size = f['x'].shape[0]
    return sample_size

def generate_data(filename, batch_size, shuffle=False):
    while True:
        for x, y in generate_data_one(filename, batch_size, shuffle):
            yield x, y

def generate_data_one(filename, batch_size, shuffle=False):
    with h5py.File(filename, 'r') as f:
        sample_size = get_dataset_size(filename)
        sample_idx = list(range(sample_size))
        if shuffle:
            random.shuffle(sample_idx)
        sample_idxs = [
            sorted(sample_idx[batch_i*batch_size:(batch_i+1)*batch_size])
            for batch_i in range(sample_size // batch_size)
        ]
        for sample_idx in sample_idxs:
            x = f['x'][sample_idx]
            y = dict(
                (k, f['y/{}'.format(k)][sample_idx])
                for k in ('mask', 'embedding')
            )
            if shuffle:
                idx = np.arange(batch_size)
                np.random.shuffle(idx)
                x = x[idx]
                y = dict((k, v[idx]) for k, v in y.items())
            yield x, y

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-i', '--train-data', type=str, required=True,
        metavar='PATH', help='Train dataset path'
    )
    parser.add_argument(
        '-o', '--output-model', type=str, required=True,
        metavar='PATH', help='Output model path'
    )
    parser.add_argument(
        '-m', '--input-model', type=str,
        metavar='PATH',
        help='Input model path (train from this model)'
    )
    parser.add_argument(
        '-d', '--embedding-dims', type=int, default=20,
        metavar='D',
        help='Dimension of embedding, ignored -m is given (default=20)'
    )
    parser.add_argument(
        '-b', '--batch-size', type=int, default=32,
        metavar='B',
        help='Batch size of train/validation'
    )
    parser.add_argument(
        '--validation-data', type=str,
        metavar='PATH', help='Validation dtaset path'
    )
    parser.add_argument(
        '--log', type=str,
        metavar='PATH', help='Log path'
    )
    parser.add_argument(
        '--stop-epoch', type=int, default=None,
        metavar='N',
        help='Train stops on this epoch (default=initial_epoch+1)'
    )
    parser.add_argument(
        '--initial-epoch', type=int, default=0,
        metavar='N',
        help='Train starts on this epoch (default=0)'
    )

    args = parser.parse_args()
    if args.stop_epoch is None:
        args.stop_epoch = args.initial_epoch + 1

    return args

if __name__ == '__main__':
    main()
