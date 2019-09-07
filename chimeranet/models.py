"""model module
"""

import json
import keras.backend as K
import keras.models
from keras.models import Model
from keras.layers import Input, Lambda, Reshape, Activation
from keras.layers import Dense, LSTM, Bidirectional
from keras.utils import CustomObjectScope
from keras.utils.io_utils import H5Dict

def loss_deepclustering(T, F, C, D):
    def _loss_deepclustering(y_true, y_pred):
        Y = K.reshape(y_true, (-1, T*F, C))
        V = K.reshape(y_pred, (-1, T*F, D))
        Dm = K.expand_dims(K.batch_dot(
            Y,
            K.sum(Y, axis=(1,)),
            axes=(2, 1)
        ))
        Dm = K.switch(Dm > 0, Dm**-0.25, K.zeros(K.shape(Dm)))
        DV, DY = Dm * V, Dm * Y
        a = K.sum(K.batch_dot(DV, DV, axes=(1, 1))**2, axis=(1, 2))
        b = K.sum(K.batch_dot(DV, DY, axes=(1, 1))**2, axis=(1, 2))
        c = K.sum(K.batch_dot(DY, DY, axes=(1, 1))**2, axis=(1, 2))
        return (a - 2*b + c)
    return _loss_deepclustering

def loss_mask(T, F, C, D):
    def _loss_mask(y_true, y_pred):
        mixture = K.expand_dims(y_true[:, :, :, C])
        mask_true = y_true[:, :, :, :C]
        return K.sum(
            K.pow((mask_true - y_pred)*mixture, 2), axis=(1, 2, 3)
        )
    return _loss_mask

def build_chimeranet_model(T, F, C, D, n_blstm_units=500, n_blstm_layers=4):
    inputs = Input(shape=(T, F), name='input')
    blstm_top = inputs
    for i in range(1, n_blstm_layers+1):
        blstm_top = Bidirectional(
            LSTM(n_blstm_units, return_sequences=True),
            name='body_blstm_{}'.format(i)
        )(blstm_top)
    body_linear = Dense(F*D, name='body_linear')(blstm_top)
    body = Reshape((T, F, D), name='body')(body_linear)

    embd_activation = Activation('tanh', name='embedding_activation')(body)
    embedding = Lambda(
        lambda x: K.l2_normalize(x, axis=-1), name='embedding'
    )(embd_activation)
    mask_slices = [
        Lambda(
            lambda x: x[:, :, f, :], name='mask_slice_{}'.format(f+1)
        )(body)
        for f in range(F)
    ]
    mask_linears = [
        Dense(
            C, activation='softmax', name='mask_linear_{}'.format(f+1)
        )(mask_slices[f])
        for f in range(F)
    ]
    mask = Lambda(lambda x: K.stack(x, axis=2), name='mask')(mask_linears)

    model = Model(inputs=inputs, outputs=[embedding, mask])
    return model

def build_chimerapp_model(T, F, C, D, n_blstm_units=500, n_blstm_layers=4):
    inputs = Input(shape=(T, F), name='input')
    blstm_top = inputs
    for i in range(1, n_blstm_layers+1):
        blstm_top = Bidirectional(
            LSTM(n_blstm_units, return_sequences=True),
            name='body_blstm_{}'.format(i)
        )(blstm_top)

    embd_linear = Dense(
        F*D, activation='tanh', name='embedding_linear'
    )(blstm_top)
    embd_reshape = Reshape(
        (T, F, D), name='embedding_reshape'
    )(embd_linear)
    embedding = Lambda(
        lambda x: x / K.expand_dims(
            K.maximum(K.sqrt(K.sum(x**2, axis=-1)), K.epsilon())
        ),
        name='embedding'
    )(embd_reshape)

    mask_linear = Dense(F*C, name='mask_linear')(blstm_top)
    mask_reshape = Reshape((T, F, C), name='mask_reshape')(mask_linear)
    mask = Activation('softmax', name='mask')(mask_reshape)
        
    model = Model(inputs=inputs, outputs=[embedding, mask])
    return model

def probe_model_shape(path):
    h5dict = H5Dict(path, mode='r')

    layers = json.loads(h5dict['model_config'])['config']['layers']
    input_layer = next(
        l for l in layers if l['name'] == 'input')
    B, T, F = input_layer['config']['batch_input_shape']
    mask_layer = next(
        l for l in layers if l['name'] == 'mask_reshape')
    T_, F_, C = mask_layer['config']['target_shape']
    embedding_layer = next(
        l for l in layers if l['name'] == 'embedding_reshape')
    T__, F__, D = embedding_layer['config']['target_shape']

    h5dict.close()
    return T, F, C, D

def load_model(path):
    shape = probe_model_shape(path)
    with CustomObjectScope({
        '_loss_deepclustering': loss_deepclustering(*shape),
        '_loss_mask': loss_mask(*shape),
    }):
        model = keras.models.load_model(path)
    return model

def to_prediction_data(y):
    pass

def from_predicted_embedding(y):
    pass

def from_predicted_mask(y):
    pass

class ChimeraNetModel:
    """ChimeraNetModel class
    """
    def __init__(self, time_frames, mel_bins, n_channels, d_embeddings):
        """
        """
        self.T = time_frames
        self.F = mel_bins
        self.C = n_channels
        self.D = d_embeddings
    def loss_deepclustering(self):
        return loss_deepclustering(self.T, self.F, self.C, self.D)

    def loss_mask(self):
        return loss_mask(self.T, self.F, self.C, self.D)

    def build_model(self, n_blstm_units=500, n_blstm_layers=4):
        return build_chimeranet_model(
            self.T, self.F, self.C, self.D, n_blstm_units, n_blstm_layers
        )

class ChimeraPPModel(ChimeraNetModel):
    def build_model(self, n_blstm_units=500, n_blstm_layers=4):
        return build_chimerapp_model(
            self.T, self.F, self.C, self.D, n_blstm_units, n_blstm_layers
        )