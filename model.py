
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Activation
from keras.layers import Permute, Reshape, Concatenate
from keras.layers import Dense, LSTM
from keras.layers import TimeDistributed, Bidirectional

def make_loss_dc(F, T, C, D):
    # y_true: CxFxT tensor, y_pred: TxFxD tensor
    def loss_dc(y_true, y_pred):
        sample_size = K.shape(y_true)[0]
        y_true = K.reshape(K.permute_dimensions(y_true, (0, 3, 2, 1)), (sample_size, F*T, C))
        y_pred = K.reshape(y_pred, (sample_size, F*T, D))
        A = K.batch_dot(y_true, K.permute_dimensions(y_true, (0, 2, 1)), axes=[2, 1])
        Ahat = K.batch_dot(y_pred, K.permute_dimensions(y_pred, (0, 2, 1)), axes=[2, 1])
        return K.sum(K.sum(K.pow(A - Ahat, 2), axis=-1), axis=-1) / (F*T)
    return loss_dc

def make_loss_mss(F, T, C, D):
    # y_true: C+1xFxT tensor, y_pred: TxFxC tensor
    def loss_msa(y_true, y_pred):
        y_true = K.permute_dimensions(y_true, (0, 3, 2, 1))
        R, S = y_true[:, :, :, :C], K.expand_dims(y_true[:, :, :, C])
        return K.sum(K.sum(K.sum(K.pow(R - y_pred*S, 2), axis=-1), axis=-1), axis=-1)
    return loss_msa

def make_loss_mmsa(F, T, C, D):
    # y_true: C+1xFxT tensor, y_pred: TxFxC tensor
    def loss_mmsa(y_true, y_pred):
        y_true = K.permute_dimensions(y_true, (0, 3, 2, 1))
        O, S = y_true[:, :, :, :C], K.expand_dims(y_true[:, :, :, C])
        return K.sum(K.sum(K.sum(K.pow((O - y_pred)*S, 2), axis=-1), axis=-1), axis=-1)
    return loss_mmsa

def build_model(F, T, C=2, D=20, N=500):
    inputs = Input(shape=(F,T))
    permute = Permute((2, 1))(inputs)
    blstm1 = Bidirectional(LSTM(N // 2, return_sequences=True))(permute)
    blstm2 = Bidirectional(LSTM(N // 2, return_sequences=True))(blstm1)
    blstm3 = Bidirectional(LSTM(N // 2, return_sequences=True))(blstm2)
    blstm4 = Bidirectional(LSTM(N // 2, return_sequences=True))(blstm3)
    blstmn = blstm4

    linear = Permute((1, 3, 2), name='Z')(
        Concatenate()([
            Lambda(lambda x: K.expand_dims(x, axis=-1), name='expand_dims_linear_{}'.format(f))(
                TimeDistributed(Dense(D))(blstmn)
            ) for f in range(F)
        ])
    )

    # TxFxD tensor
    embd = Permute((1, 2, 3), name='V')(
        Lambda(lambda x: K.l2_normalize(x, axis=-1))(
            TimeDistributed(Activation('tanh'))(linear)
        )
    )

    # TxFxC tensor
    mask = Permute((1, 3, 2), name='M')(Concatenate()([
        Lambda(lambda x: K.expand_dims(x, axis=-1), name='expand_dims_mask_{}'.format(f))(
            TimeDistributed(Dense(C, activation='softmax'))(
                Lambda(lambda x: x[:, :, f, :], name='split_mask_{}'.format(f))(linear)
            )
        ) for f in range(F)
    ]))
    
    model = Model(inputs=inputs, outputs=[embd, mask])
    return model
