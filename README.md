# ChimeraNet
An implementation of music separation model by Luo et.al.

### Getting started

##### Sample separation task with pretrained model

0. Prepare .wav files to separate.
1. Install library
`pip install git+https://github.com/arity-r/ChimeraNet`

2. Download [pretrained model](https://github.com/arity-r/chimeranet-models/raw/master/utterance-music/bisep_120.hdf5).
3. Download [sample script](https://raw.githubusercontent.com/arity-r/ChimeraNet/master/scripts/chimeranet-separate.py).
4. Run script

```
python chimeranet-separate.py -i ${input_dir}/*.wav \
    -m model.hdf5 \
    --replace-top-directory ${output_dir}
```

Output in nutshell
* the format of filename is `${input_file}_{embd,mask}_ch[12].wav`.
* `embd` and `mask` indicates that it was inferred from deep clustering and mask respectively.
* `ch1` and `ch2` are voice and music channel respectively.

##### Train and separation examples

See [Example section on ChimeraNet documentation](https://arity-r.github.io/ChimeraNet/examples.html).

### Install

##### Requirements

* keras
* one of keras' backends (i.e. TensorFlow, CNTK, Theano)
* sklearn
* librosa
* soundfile

##### Instructions

1. Run `pip install git+https://github.com/arity-r/ChimeraNet` or
any python package installer.
(Currently, `ChimeraNet` is not in PyPI.)
2. Install keras' backend if the environment does not have any.
Install `tensorflow` if unsure.

### See also

* [Luo et.al.](https://arxiv.org/abs/1611.06265)
* [Hershey et.al.](https://arxiv.org/abs/1508.04306)
* [Original ChimeraNet site](http://danetapi.com/chimera)
