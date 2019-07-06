# ChimeraNet
An implementation of music separation model by Luo et.al.

### Sample separation tasks and model

See `sample` folder (TBA).

Also, see [Example section on ChimeraNet documentation](https://arity-r.github.io/robust-graph/examples.html).

### Install

##### Requirements

* keras
* one of keras' backends (i.e. TensorFlow, CNTK, Theano)
* sklearn
* librosa
* soundfile
* ffmpeg (in PATH, required in VoxCeleb2Loader)

##### Instructions

1. Run `pip install git+https://github.com/arity-r/ChimeraNet` or
any python package installer.
(Currently, `ChimeraNet` is not in PyPI.)
2. Install keras' backend if the environment does not have any.
Install `tensorflow` if unsure.
3. (Optional) Download ffmpeg binary and place it on PATH.

### See also

* [Luo et.al.](https://arxiv.org/abs/1611.06265)
* [Hershey et.al.](https://arxiv.org/abs/1508.04306)
* [Original ChimeraNet site](http://danetapi.com/chimera)
