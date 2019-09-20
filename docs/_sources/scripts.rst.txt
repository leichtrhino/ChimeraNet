Scripts
=======

chimeranet-prepare.py
---------------------

::

  usage: chimeranet-prepare.py [-h] [--c1 PATH [TAG]] [--c2 PATH [TAG]] -o PATH
                               -n N [-t T] [-f F] [--sr N] [--n-fft F]
                               [--hop-length N] [--save-audio]
                               [--sync | --async]

  optional arguments:
    -h, --help            show this help message and exit
    --c1 PATH [TAG]       Channel 1 information. PATH and TAG are csv.
    --c2 PATH [TAG]       Channel 2 information. PATH and TAG are csv.
    -o PATH, --output PATH
                          Output path
    -n N, --sample-size N
                          Sample size
    -t T, --time T        Window length in second (default=2.)
    -f F, --freq-bins F   Number of mel-spectrogram bins (default=150)
    --sr N                Sampling rate (default=44100)
    --n-fft F             FFT window size (default=2048)
    --hop-length N        Hop length on STFT (default=512)
    --save-audio          Save audio to "audio" dataset
    --sync                Synchronized dataset
    --async               Asynchronized dataset

chimeranet-show-data.py
-----------------------

::

  usage: chimeranet-show-data.py [-h] -i PATH [--output-plot FORMATTED STRING]
                                 [--write-audio]
                                 [--output-audio FORMATTED STRING] [-d DIR]
                                 [--sr N] [--channel-name [NAME [NAME ...]]]

  optional arguments:
    -h, --help            show this help message and exit
    -i PATH, --input PATH
    --output-plot FORMATTED STRING
                          Formatted string as output spectrogram plot prefix
                          (e.g. "{index:}.png").
    --write-audio         Enable output audio if exists
    --output-audio FORMATTED STRING
                          Formatted string as output audio path (e.g.
                          "{index:}_{channel:}.wav" (default)).
    -d DIR, --output-directory DIR
                          If specified, add it as top directory of "--output-
                          audio"
    --sr N                Sampling rate (default=44100)
    --channel-name [NAME [NAME ...]]
                          Channel names show on output audio and/or plot.

chimeranet-train.py
-------------------

::

  usage: chimeranet-train.py [-h] -i PATH -o PATH [-m PATH] [-d D] [-b B]
                             [--validation-data PATH] [--log PATH]
                             [--stop-epoch N] [--initial-epoch N]
  
  optional arguments:
    -h, --help            show this help message and exit
    -i PATH, --train-data PATH
                          Train dataset path
    -o PATH, --output-model PATH
                          Output model path
    -m PATH, --input-model PATH
                          Input model path (train from this model)
    -d D, --embedding-dims D
                          Dimension of embedding, ignored -m is given
                          (default=20)
    -b B, --batch-size B  Batch size of train/validation
    --validation-data PATH
                          Validation dtaset path
    --log PATH            Log path
    --stop-epoch N        Train stops on this epoch (default=initial_epoch+1)
    --initial-epoch N     Train starts on this epoch (default=0)


chimeranet-separate.py
----------------------

::

  usage: chimeranet-separate.py [-h] -m PATH -i [PATH [PATH ...]]
                                [-o FORMATTED STRING] [-d DIR] [--sr N]
                                [--n-fft F] [--hop-length N] [--duration T]
                                [--replace-top-directory DIR]
                                [--plot-spectrograms]
                                [--disable-embedding-inference]
                                [--disable-mask-output]
                                [--output-plot FORMATTED STRING]
                                [--channel-name [NAME [NAME ...]]]
                                [--embedding-inference-name NAME]
                                [--mask-inference-name NAME]
  
  optional arguments:
    -h, --help            show this help message and exit
  
  basic arguments:
    -m PATH, --model-path PATH
    -i [PATH [PATH ...]], --input-audio [PATH [PATH ...]]
    -o FORMATTED STRING, --output-audio FORMATTED STRING
                          Formatted string as output audio path (e.g.
                          "{input:}_{infer:}_{channel:}.wav" (default)).
    -d DIR, --output-directory DIR
                          If specified, add it as top directory of "--output-
                          audio"
  
  audio arguments:
    --sr N                Sampling rate (default=16000)
    --n-fft F             FFT window size (default=512)
    --hop-length N        Hop length on STFT (default=128)
    --duration T          Audio duration in seconds
  
  advanced output:
    --replace-top-directory DIR
                          If specified, replace top directory of "--output-
                          audio" with it
    --plot-spectrograms   Enable output spectrograms
    --disable-embedding-inference
                          Disable embedding inference
    --disable-mask-output
                          Disable output from mask inference
    --output-plot FORMATTED STRING
                          Formatted string as output spectrogram plot prefix
                          (e.g. "{input:}.png").
    --channel-name [NAME [NAME ...]]
                          Channel names show on output audio and/or plot.
    --embedding-inference-name NAME
                          Inference name of embedding
    --mask-inference-name NAME
                          Inference name of embedding
