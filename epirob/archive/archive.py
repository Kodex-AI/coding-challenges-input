import os
import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Reshape, Conv2D, Conv2DTranspose, concatenate, MaxPool2D
from keras.optimizers import Adam
from learning import configure_keras
from scipy.stats import ttest_ind
from image_warp import floor, ceil
from util import load_data


# from learning.py
def build_model_multimodal(input_dim=4, pixels=128):
    print 'building model...'
    input_angles = Input(shape=(input_dim,))
    angles = Dense(units=512, activation='relu')(input_angles)
    angles = Dense(units=512, activation='relu')(angles)
    angles = Dense(units=1024, activation='relu')(angles)
    angles = Dense(units=1024, activation='relu')(angles)
    angles = Dense(units=pixels ** 2 / 2, activation='relu')(angles)

    input_image = Input(shape=(pixels, pixels, 1))
    image = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(input_image)
    image = MaxPool2D(padding='same')(image)
    image = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(image)
    image = MaxPool2D(padding='same')(image)
    image = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(image)
    image = MaxPool2D(padding='same')(image)
    image = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(image)
    image = MaxPool2D(padding='same')(image)
    image = Reshape(target_shape=(8 * 8 * 128,))(image)

    concat = concatenate([angles, image])
    concat = Reshape(target_shape=(8, 8, 256))(concat)

    conv = upconv_conv_functional(concat, 256)
    conv = upconv_conv_functional(conv, 92)
    conv = upconv_conv_functional(conv, 48)

    output = Conv2DTranspose(filters=1, kernel_size=4, strides=2, padding='same')(conv)

    model = keras.models.Model(inputs=[input_angles, input_image], outputs=output)
    model.compile(optimizer=Adam(epsilon=1e-6), loss='mean_squared_error')
    model.summary()
    return model


# from learning.py
def upconv_conv_functional(last_layer, filters):
    conv = Conv2DTranspose(filters=filters, kernel_size=4, strides=2, padding='same', activation='relu')(last_layer)
    conv = Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu')(conv)
    return conv


# from learning.py
def train_model_multimodal(file_names, name, model, epochs):
    configure_keras()
    all_input_positions, all_input_images, all_output_images = None, None, None
    for i, file_name in enumerate(file_names):
        input_positions, input_images, output_images = load_data_multimodal_multiple(file_name)
        if i == 0:
            all_input_positions = np.asarray(input_positions)
            all_input_images = np.asarray(input_images)
            all_output_images = np.asarray(output_images)
        else:
            all_input_positions = np.vstack((all_input_positions, input_positions))
            all_input_images = np.vstack((all_input_images, input_images))
            all_output_images = np.vstack((all_output_images, output_images))
    all_input_images = np.expand_dims(all_input_images, axis=-1)
    all_output_images = np.expand_dims(all_output_images, axis=-1)
    callbacks = [keras.callbacks.ModelCheckpoint('data/models/' + name + '_{epoch:02d}-{loss:.2f}.h5')]
    model.fit([all_input_positions, all_input_images], all_output_images, callbacks=callbacks, batch_size=128,
              epochs=epochs, verbose=2, shuffle=True)


# from sense_of_agency.py
def unexpected_difference(flow, frame_1, frame_2, threshold, debug=False):
    shape = (flow.shape[0], flow.shape[1])
    difference = np.abs(frame_1 - frame_2)
    unexpected_movement_mask = np.ones_like(difference, dtype=bool)
    flow_intensity = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2).squeeze()
    for i in range(shape[0]):
        for j in range(shape[1]):
            if flow_intensity[i, j] > threshold:
                unexpected_movement_mask[i, j] = 0
                target_i = i + int(round(flow[i, j, 0]))
                target_j = j + int(round(flow[i, j, 1]))
                target_i = min(target_i, shape[0] - 1)
                target_j = min(target_j, shape[1] - 1)
                unexpected_movement_mask[target_i, target_j] = 0

    if debug:

        def imshow(img):
            plt.imshow(img, cmap='gray')
            plt.colorbar()
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])

        plt.subplot(2, 2, 1)
        imshow(frame_1)
        plt.subplot(2, 2, 2)
        imshow(frame_2)
        plt.subplot(2, 2, 3)
        # plt.imshow(flow_to_color(flow, img_format='rgb'))
        imshow(difference)
        plt.subplot(2, 2, 4)
        imshow(unexpected_movement_mask)
        plt.show()

    return np.mean(difference * unexpected_movement_mask)


# from sense_of_agency.py
def unexpectedness_vector(model, positions, images, threshold, debug=False):
    unexpectedness = []

    for i in range(len(positions) - 1):
        frame_1 = images[i]
        frame_2 = images[i + 1]

        frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_RGB2GRAY)
        frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_RGB2GRAY)

        flow = model.predict(positions[i].reshape((1, 8))).squeeze()
        unexpectedness.append(unexpected_difference(flow, frame_1, frame_2, threshold, debug=debug))

    return unexpectedness


# from sense_of_agency.py
def one_flow_trial(model, trial, agency_dir='data/images/agency/', threshold=5, debug=False):
    shape = model.layers[-1].output_shape[1:-1]

    agency_file_names = os.listdir(agency_dir)
    file_names_a = [file_name for file_name in agency_file_names if '_{}a_'.format(trial) in file_name]
    file_names_b = [file_name for file_name in agency_file_names if '_{}b_'.format(trial) in file_name]

    file_name_a = file_names_a[0] if len(file_names_a) == 1 else [name for name in file_names_a if 'to' in name][0]
    file_name_b = file_names_b[0] if len(file_names_b) == 1 else [name for name in file_names_b if 'to' in name][0]

    positions_a, images_a = load_data(8, agency_dir + file_name_a, shape, crop=True)
    positions_b, images_b = load_data(8, agency_dir + file_name_b, shape, crop=True)

    unexpectedness_1 = unexpectedness_vector(model, positions_a, images_a, threshold, debug=debug)
    unexpectedness_2 = unexpectedness_vector(model, positions_b, images_b, threshold, debug=debug)

    t, p = ttest_ind(unexpectedness_1, unexpectedness_2)

    print('trial', trial)
    print('mean error a:', np.mean(unexpectedness_1))
    print('mean error b:', np.mean(unexpectedness_2))
    print('p value:', '{:0.2e}'.format(p))
    print()


# from analysis.py
class AnalysisMultiModal:

    def __init__(self, model_file_name):
        self.model = keras.models.load_model(model_file_name)

    def predict_sequence(self, file_name, first_index=None, length=5):
        input_deltas, input_images, output_images = load_data_multimodal(file_name)

        if first_index is None:
            first_index = np.random.randint(len(input_deltas) - length)

        ax = plt.subplot2grid((2, length + 1), (0, 0))
        self.imshow(ax, output_images[first_index])
        plt.xlabel('Img(0)')
        plt.ylabel('originals')

        ax = plt.subplot2grid((2, length + 1), (1, 0))
        self.imshow(ax, np.zeros((2, 2)))
        ax.set_ylabel('predictions')

        for i in range(length):
            original = output_images[first_index + i]
            delta_pos = input_deltas[first_index + 1].reshape(1, 4)
            input_image = input_images[first_index + i].reshape(1, 128, 128, 1)
            prediction = self.model.predict([delta_pos, input_image])

            ax = plt.subplot2grid((2, length + 1), (0, i + 1))
            self.imshow(ax, original)
            ax.set_xlabel('Img({})'.format(i + 1))

            ax = plt.subplot2grid((2, length + 1), (1, i + 1))
            self.imshow(ax, prediction)
            ax.set_xlabel('Img*({})'.format(i + 1))

        plt.show()

    def predict_any_two(self, file_name, distance=1, first_index=None):
        positions, images = load_data(4, file_name, 128, 1)

        if first_index is None:
            idx_1 = np.random.randint(len(positions) - distance)
        else:
            idx_1 = first_index
        idx_2 = idx_1 + distance

        ax = plt.subplot2grid((2, 2), (0, 0))
        self.imshow(ax, images[idx_1])
        ax.set_xlabel('Img(0)')

        ax = plt.subplot2grid((2, 2), (0, 1))
        self.imshow(ax, images[idx_2])
        ax.set_xlabel('Img(1)')

        ax = plt.subplot2grid((2, 2), (1, 1))
        delta_pos = (positions[idx_2] - positions[idx_1]).reshape(1, 4)
        prediction = self.model.predict([delta_pos, images[idx_1].reshape(1, 128, 128, 1)])
        self.imshow(ax, prediction)
        ax.set_xlabel('Img*(1)')

        plt.suptitle('Angle Delta: ' + str(delta_pos))
        plt.show()

    @staticmethod
    def imshow(ax, img, cmap='gray'):
        ax.imshow(img.squeeze(), cmap=cmap)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])


# from image_warp.py
def forward_warp_interpolating_old(im, flow, method='nearest'):
    height, width, channels = im.shape
    i_s, j_s = [], []
    new_is, new_js = [], []
    values = [[], [], []]
    for i in range(height):
        for j in range(width):
            i_s.append(i)
            j_s.append(j)
            new_is.append(i + flow[i, j, 1])
            new_js.append(j + flow[i, j, 0])
            for channel in range(channels):
                values[channel].append(im[i, j, channel])
    warped = []

    i_s = np.asarray(i_s)
    j_s = np.asarray(j_s)
    new_is = np.array(new_is)
    new_js = np.array(new_js)
    values = [np.asarray(vs) for vs in values]

    for channel in range(channels):
        warped.append(griddata(points=(new_is, new_js), values=values[channel], xi=(i_s, j_s), method=method,
                               fill_value=im[:, :, channel].mean())
                      .reshape(height, width))
    return np.dstack(warped)


# from image_warp.py
def forward_warp_rounding(im, flow):
    height, width, channels = im.shape
    # warped = np.copy(im)
    warped = np.zeros_like(im)
    for i in range(height):
        for j in range(width):
            new_i = int(np.round(i + flow[i, j, 1]))
            new_j = int(np.round(j + flow[i, j, 0]))
            if 0 <= new_i < height and 0 <= new_j < width:
                warped[new_i, new_j, :] = im[i, j, :]
    return warped


# from image_warp.py
def forward_warp_sampling(im, flow):
    height, width, channels = im.shape
    warped = np.zeros_like(im, dtype=float)
    for i in range(height):
        for j in range(width):
            value = im[i, j, :]
            new_i = i + flow[i, j, 1]
            new_j = j + flow[i, j, 0]
            fraction_i = new_i - floor(new_i)
            fraction_j = new_j - floor(new_j)

            if 0 <= floor(new_i) < height and 0 <= floor(new_j) < width:
                warped[floor(new_i), floor(new_j), :] += (1 - fraction_i) * (1 - fraction_j) * value
            if 0 <= floor(new_i) < height and 0 <= ceil(new_j) < width:
                warped[floor(new_i), ceil(new_j), :] += (1 - fraction_i) * fraction_j * value
            if 0 <= ceil(new_i) < height and 0 <= floor(new_j) < width:
                warped[ceil(new_i), floor(new_j), :] += fraction_i * (1 - fraction_j) * value
            if 0 <= ceil(new_i) < height and 0 <= ceil(new_j) < width:
                warped[ceil(new_i), ceil(new_j), :] += fraction_i * fraction_j * value

    return warped


# from image_warp.py
def backward_warp(im, flow):
    """Performs a backward warp of an image using the predicted flow.

    Args:
        im: Batch of images. [num_batch, height, width, channels]
        flow: Batch of flow vectors. [num_batch, height, width, 2]
    Returns:
        warped: transformed image of the same shape as the input image.
    """
    with tf.variable_scope('image_warp'):

        num_batch, height, width, channels = tf.unstack(tf.shape(im))
        max_x = tf.cast(width - 1, 'int32')
        max_y = tf.cast(height - 1, 'int32')
        zero = tf.zeros([], dtype='int32')

        # We have to flatten our tensors to vectorize the interpolation
        im_flat = tf.reshape(im, [-1, channels])
        flow_flat = tf.reshape(flow, [-1, 2])

        # Floor the flow, as the final indices are integers
        # The fractional part is used to control the bilinear interpolation.
        flow_floor = tf.to_int32(tf.floor(flow_flat))
        bilinear_weights = flow_flat - tf.floor(flow_flat)

        # Construct base indices which are displaced with the flow
        pos_x = tf.tile(tf.range(width), [height * num_batch])
        grid_y = tf.tile(tf.expand_dims(tf.range(height), 1), [1, width])
        pos_y = tf.tile(tf.reshape(grid_y, [-1]), [num_batch])

        x = flow_floor[:, 0]
        y = flow_floor[:, 1]
        xw = bilinear_weights[:, 0]
        yw = bilinear_weights[:, 1]

        # Compute interpolation weights for 4 adjacent pixels
        # expand to num_batch * height * width x 1 for broadcasting in add_n below
        wa = tf.expand_dims((1 - xw) * (1 - yw), 1)  # top left pixel
        wb = tf.expand_dims((1 - xw) * yw, 1)        # bottom left pixel
        wc = tf.expand_dims(xw * (1 - yw), 1)        # top right pixel
        wd = tf.expand_dims(xw * yw, 1)              # bottom right pixel

        x0 = pos_x + x
        x1 = x0 + 1
        y0 = pos_y + y
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        dim1 = width * height
        batch_offsets = tf.range(num_batch) * dim1
        base_grid = tf.tile(tf.expand_dims(batch_offsets, 1), [1, dim1])
        base = tf.reshape(base_grid, [-1])

        base_y0 = base + y0 * width
        base_y1 = base + y1 * width
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        warped_flat = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        warped = tf.reshape(warped_flat, [num_batch, height, width, channels])

        return warped


# from learning.py
def train_model_multiple_files(file_names, name, model, input_dim, epochs=10):
    all_positions, all_images = None, None
    for i, file_name in enumerate(file_names):
        positions, images = load_data(input_dim, file_name, 128, 1, convert=False, resize=True)
        if i == 0:
            all_positions = positions
            all_images = images
        else:
            all_positions = np.vstack((all_positions, positions))
            all_images = np.vstack((all_images, images))
    train_model(name, model, all_positions, all_images, epochs=epochs)


# from unit_tests.py
def test_warp_sampling(self):
    expected = np.ones((3, 3))
    expected[2, 0] = 0
    expected[0, 1] = 0.3 + 1
    expected[0, 2] = 0.7
    expected[1, 1] = 2.7 + 1
    expected[1, 2] = 6.3 + 1
    warped = image_warp.forward_warp_sampling(self.img, self.flow).squeeze()
    np.testing.assert_array_almost_equal(expected, warped)


# from object_permanence.py
def enhance_with_flow_predictor_old(positions, images, model, threshold=0.5, update_background=False):
    input_dim = positions.shape[1]
    background = images[0].squeeze()
    enhanced_images = []
    predictions = []
    masks = []
    for i, image in enumerate(images):
        position = positions[i].reshape((1, input_dim))
        flow = model.predict(position).squeeze()
        predictions.append(flow)
        flow_intensity = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        mask = flow_intensity > threshold
        background_mask = flow_intensity > threshold    # todo: fine-tune
        masks.append(mask)
        mask = np.dstack((mask,) * 3)
        background_mask = np.dstack((background_mask,) * 3)
        not_mask = np.logical_not(mask)
        not_mask_background = np.logical_not(background_mask)
        enhanced = background * mask + image.squeeze() * not_mask
        if update_background:
            background = background * background_mask + image.squeeze() * not_mask_background
        enhanced_images.append(enhanced)
    return np.asarray(enhanced_images), np.asarray(predictions), np.asarray(masks)


# from util.py
def load_data_multimodal(file_name):
    input_images = []
    input_positions = []
    output_images = []
    print 'loading training data...'
    with gzip.open(file_name, 'rb') as memory_file:
        memories = cPickle.load(memory_file)
        print 'converting training data...'
        previous_position = None
        previous_image = None
        for i, memory in enumerate(memories):
            if i > 0:
                current_image = memory['image']
                output_images.append(current_image)
                current_position = memory['sensor_angles']
                delta_pos = [current_position[k] - previous_position[k] for k in range(4)]
                input_positions.append(delta_pos)
                input_images.append(previous_image)
            previous_position = memory['sensor_angles']
            previous_image = memory['image']
    input_positions = np.asarray(input_positions)
    input_images = np.asarray(input_images)
    output_images = np.asarray(output_images)
    output_images = output_images.reshape((len(output_images), 128, 128, 1))
    return input_positions, input_images, output_images


# from util.py
def load_data_multimodal_multiple(file_name, max_distance=20):
    input_images = []
    input_positions = []
    output_images = []
    print 'loading training data...'
    with gzip.open(file_name, 'rb') as memory_file:
        memories = cPickle.load(memory_file)
        print 'converting training data...'
        for i, memory in enumerate(memories):
            current_image = memory['image']
            current_position = memory['sensor_angles']
            for j in range(max_distance + 1)[::2]:
                if i + j >= len(memories):
                    break
                next_image = memories[i + j]['image']
                next_position = memories[i + j]['sensor_angles']
                delta_pos = [next_position[k] - current_position[k] for k in range(4)]
                input_images.append(current_image)
                input_positions.append(delta_pos)
                output_images.append(next_image)
    return input_positions, input_images, output_images