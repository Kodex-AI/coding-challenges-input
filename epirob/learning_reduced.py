# author: Claus Lang
# email: claus.lang@bccn-berlin.de
import datetime

import keras
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import Dense, Reshape, Conv2D, Conv2DTranspose, PReLU, Input
from keras.optimizers import Adam
from keras.models import Model

from util import load_data, threshold_flows


def build_model(input_dim=4, output_channels=1):
    print 'building model...'
    model = keras.models.Sequential()

    model.add(Dense(units=512, input_dim=input_dim, activation='relu'))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=8 * 8 * 256, activation='relu'))
    model.add(Reshape(target_shape=(8, 8, 256)))

    upconv_conv(model, 256)
    upconv_conv(model, 92)
    upconv_conv(model, 48)

    # upconv4
    model.add(Conv2DTranspose(filters=output_channels, kernel_size=4, strides=2, padding='same'))

    model.compile(optimizer=Adam(epsilon=1e-6), loss='mean_squared_error')
    model.summary()
    return model


def build_flow_model(input_dim=8):
    print('building model...')
    model = keras.models.Sequential()

    model.add(Dense(units=512, input_dim=input_dim, activation='relu'))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=6 * 8 * 256, activation='relu'))
    model.add(Reshape(target_shape=(6, 8, 256)))

    upconv_conv(model, 256)
    upconv_conv(model, 128)
    upconv_conv(model, 92)
    upconv_conv(model, 48)

    # upconv4
    model.add(Conv2DTranspose(filters=2, kernel_size=4, strides=2, padding='same'))

    # model.compile(optimizer=Adam(epsilon=1e-6), loss='mean_squared_error')
    model.compile(optimizer=Adam(epsilon=1e-6), loss='mean_absolute_error')
    model.summary()
    return model
