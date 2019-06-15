
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Reshape, Activation
from keras.layers import Dense, LSTM, Bidirectional

def loss_deepclustering(y_true, y_pred):
    C, (batch_size, T, F, D) = K.int_shape(y_true)[-1], K.int_shape(y_pred)
    if C is None:
        C = 1
    y_true_flatten = K.reshape(y_true, (-1, T*F, C))
    y_pred_flatten = K.reshape(y_pred, (-1, T*F, D))
    A = K.batch_dot(
        y_true_flatten,
        K.permute_dimensions(y_true_flatten, (0, 2, 1)), axes=[2, 1]
    )
    Ahat = K.batch_dot(
        y_pred_flatten,
        K.permute_dimensions(y_pred_flatten, (0, 2, 1)), axes=[2, 1]
    )
    return K.sum(K.pow(A - Ahat, 2), axis=(1, 2)) / K.cast(F*T, K.floatx())

def loss_mask(y_true, y_pred):
    batch_size, T, F, C = K.int_shape(y_pred)
    mixture = K.expand_dims(y_true[:, :, :, C])
    return K.sum(K.pow((y_true[:, :, :, :C] - y_pred)*mixture, 2), axis=(1, 2, 3))

def build_model(T, F, C, D, n_blstm_units=500, n_blstm_layers=4):
    inputs = Input(shape=(T, F), name='input')
    blstm_top = inputs
    for i in range(1, n_blstm_layers+1):
        blstm_top = Bidirectional(
            LSTM(n_blstm_units, return_sequences=True),
            name='body_blstm_{}'.format(i)
        )(blstm_top)
    body = Reshape((T, F, D), name='body_reshape')(Dense(F*D, name='body_linear')(blstm_top))

    embedding = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='embedding')(
        Activation('tanh', name='embedding_activation')(body)
    )
    mask = Lambda(lambda x: K.stack(x, axis=2), name='mask')([
        Dense(C, activation='softmax', name='mask_linear_{}'.format(f+1))(
            Lambda(lambda x: x[:, :, f, :], name='mask_slice_{}'.format(f+1))(body)
        ) for f in range(F)
    ])
    
    model = Model(inputs=inputs, outputs=[embedding, mask])
    return model
