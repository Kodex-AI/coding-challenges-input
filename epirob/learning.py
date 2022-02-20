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


def build_flow_model_simple(input_dim=8, activation='relu'):
    print('building model...')
    model = keras.models.Sequential()

    model.add(Dense(units=6 * 8 / 2, input_dim=input_dim, activation=activation))
    model.add(Dense(units=6 * 8 * 4, activation=activation))
    model.add(Dense(units=6 * 8 * 32, activation=activation))
    model.add(Reshape(target_shape=(6, 8, 32)))

    upconv_conv(model, 32)
    upconv_conv(model, 16)
    upconv_conv(model, 8)
    upconv_conv(model, 4)

    # upconv4
    model.add(Conv2DTranspose(filters=2, kernel_size=4, strides=2, padding='same'))

    model.compile(optimizer=Adam(epsilon=1e-6), loss='mean_absolute_error')
    model.summary()
    return model


def build_flow_model_simple_prelu(input_dim=8):
    print('building model...')
    model = keras.models.Sequential()

    model.add(Dense(units=6 * 8 / 2, input_dim=input_dim))
    model.add(PReLU())
    model.add(Dense(units=6 * 8 * 4))
    model.add(PReLU())
    model.add(Dense(units=6 * 8 * 32))
    model.add(PReLU())
    model.add(Reshape(target_shape=(6, 8, 32)))

    upconv_conv_prelu(model, 32)
    upconv_conv_prelu(model, 16)
    upconv_conv_prelu(model, 8)
    upconv_conv_prelu(model, 4)

    # upconv4
    model.add(Conv2DTranspose(filters=2, kernel_size=4, strides=2, padding='same'))

    model.compile(optimizer=Adam(epsilon=1e-6), loss='mean_absolute_error')
    model.summary()
    return model


def build_flow_model_simple_streams_prelu():
    print('building model')
    pos_input = Input((4,))
    pos_stream = Dense(units=4)(pos_input)
    pos_stream = PReLU()(pos_stream)
    pos_stream = Dense(units=6 * 8)(pos_stream)
    pos_stream = PReLU()(pos_stream)
    pos_stream = Dense(units=6 * 8 * 4)(pos_stream)
    pos_stream = PReLU()(pos_stream)

    delta_input = Input((4,))
    delta_stream = Dense(units=4)(delta_input)
    delta_stream = PReLU()(delta_stream)
    delta_stream = Dense(units=6 * 8)(delta_stream)
    delta_stream = PReLU()(delta_stream)
    delta_stream = Dense(units=6 * 8 * 4)(delta_stream)
    delta_stream = PReLU()(delta_stream)

    stream = keras.layers.concatenate([pos_stream, delta_stream])
    stream = Dense(units=6 * 8 * 64)(stream)
    stream = PReLU()(stream)
    stream = Reshape(target_shape=(6, 8, 64))(stream)

    stream = upconv_conv_prelu_functional(stream, 64)
    stream = upconv_conv_prelu_functional(stream, 32)
    stream = upconv_conv_prelu_functional(stream, 16)
    stream = upconv_conv_prelu_functional(stream, 8)
    stream = upconv_conv_prelu_functional(stream, 4)
    output = Conv2DTranspose(filters=2, kernel_size=4, strides=2, padding='same')(stream)

    model = Model(inputs=[pos_input, delta_input], outputs=[output])
    model.compile(optimizer=Adam(epsilon=1e-6), loss='mean_absolute_error')
    model.summary()
    return model


def upconv_conv_prelu_functional(stream, filters):
    stream = Conv2DTranspose(filters=filters, kernel_size=4, strides=2, padding='same')(stream)
    stream = PReLU()(stream)
    stream = Conv2D(filters=filters, kernel_size=3, padding='same')(stream)
    stream = PReLU()(stream)
    return stream


def upconv_conv_prelu(model, filters):
    # upconv
    model.add(Conv2DTranspose(filters=filters, kernel_size=4, strides=2, padding='same'))
    model.add(PReLU())
    # conv
    model.add(Conv2D(filters=filters, kernel_size=3, padding='same'))
    model.add(PReLU())


def upconv_conv(model, filters):
    # upconv
    model.add(Conv2DTranspose(filters=filters, kernel_size=4, strides=2, padding='same', activation='relu'))
    # conv
    model.add(Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu'))


# def configure_keras(gpu_fraction=0.3):
def configure_keras(gpu_fraction=0.7):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
    set_session(tf.Session(config=config))


def train_model(name, model, positions, images, epochs=10):
    configure_keras()
    print 'training model...'
    callbacks = [keras.callbacks.ModelCheckpoint('data/models/' + name + '_{epoch:02d}-{loss:.4f}.h5')]
    model.fit(positions, images, callbacks=callbacks, batch_size=128, epochs=epochs, verbose=2, shuffle=True)


def train_stream_model(name, model, positions, images, epochs=10):
    configure_keras()
    print 'training model...'
    callbacks = [keras.callbacks.ModelCheckpoint('data/models/' + name + '_{epoch:02d}-{loss:.4f}.h5')]
    positions = [positions[:, :4], positions[:, 4:]]
    model.fit(positions, images, callbacks=callbacks, batch_size=128, epochs=epochs, verbose=2, shuffle=True)


def main():
    print 'started at', datetime.datetime.now()

    positions, _ = load_data(8, 'data/images/grey400_bw.pkl')
    _, images = load_data(4, 'data/images/grey400_flow.pkl', resize=False, convert=False)
    assert len(positions) == len(images)

    model = build_flow_model_simple_prelu()
    train_model('test', model, positions, images, epochs=2)


if __name__ == '__main__':
    main()
