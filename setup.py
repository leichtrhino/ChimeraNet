#!/usr/bin/env python

from distutils.core import setup

setup(
    name='ChimeraNet',
    version='1.0',
    description='',
    author='',
    author_email='',
    url='https://github.com/arity-r/ChimeraNet',
    packages=['chimeranet'],
    package_dir={'chimeranet': 'chimeranet'},
    package_data={'chimeranet': ['model/*.hd5']},
    scripts=['scripts/chimeranet', 'scripts/chimeranet-train'],
)
