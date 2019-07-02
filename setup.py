#!/usr/bin/env python

from setuptools import setup

setup(
    name='ChimeraNet',
    version='0.1',
    description='An implementation of music separation model by Luo et.al.',
    author='',
    author_email='',
    url='https://github.com/arity-r/ChimeraNet',
    packages=['chimeranet'],
    python_requires='~=3.6',
    # TODO: specify version
    install_requires=[
        'h5py',
        'keras',
        'scikit-learn',
        'librosa',
        'pysoundfile',
    ],
    package_dir={'chimeranet': 'chimeranet'},
    package_data={'chimeranet': ['models/*.hd5']},
    scripts=[
        'scripts/chimeranet.py',
        'scripts/chimeranet-prepare.py',
        'scripts/chimeranet-train.py',
        'scripts/chimeranet-separate.py',
    ],
)
