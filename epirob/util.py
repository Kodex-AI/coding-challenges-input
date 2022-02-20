# author: Claus Lang
# email: claus.lang@bccn-berlin.de
import gzip
import cPickle
import cv2
import re
import numpy as np
import PIL


COMPLETE_RANGES = {'HeadYaw': {'min': -2.0857, 'max': 2.0857},
                   'HeadPitch': {'min': -0.672, 'max': 0.5149},
                   'LShoulderPitch': {'min': -2.0857, 'max': 2.0857},
                   'LShoulderRoll': {'min': -0.3142, 'max': 1.3265},
                   'LElbowYaw': {'min': -2.0857, 'max': 2.0857},
                   'LElbowRoll': {'min': -1.5446, 'max': -0.0349},
                   'LWristYaw': {'min': -1.8238, 'max': 1.8238}}

BIG_RANGES = {'LShoulderPitch': {'min': -1.9, 'max': 0.9},
              'LShoulderRoll': {'min': -0.3142, 'max': 1.1},
              'LElbowRoll': {'min': -1.5446, 'max': -0.0349},
              'LElbowYaw': {'min': -2.0857, 'max': 2.0857}}

RANGES = {'LShoulderPitch': {'min': -1.3, 'max': 0.5},
          'LShoulderRoll': {'min': -0.3142, 'max': 0.5},
          'LElbowRoll': {'min': -1.5446, 'max': -0.0349},
          'LElbowYaw': {'min': -0.6, 'max': 2.0857}}

IP = '192.168.26.51'
PORT = 9559


def load_data(dim, file_name, shape=(320, 240), channels=1, convert=False, resize=False, crop=False):
    if dim == 4:
        return load_data_4_dim(file_name, shape, channels, convert, resize, crop)
    if dim == 8:
        return load_data_8_dim(file_name, shape, channels, convert, resize, crop)
    raise NotImplementedError('Only dimensions 4 and 8 supported, but not {}'.format(dim))


def load_image(memory):
    if len(memory['image']) == 12:
        image = memory['image'][6]
        image = PIL.Image.frombytes('RGB', (320, 240), image)
        return np.array(image)
    else:
        return memory['image']


def load_data_4_dim(file_name, shape, channels, convert, resize, crop):
    images = []
    positions = []
    print 'loading training data...'
    with gzip.open(file_name, 'rb') as memory_file:
        memories = cPickle.load(memory_file)
        print 'converting training data...'
        for i, memory in enumerate(memories):
            image = load_image(memory)
            image = preprocess_image(image, shape, channels, convert, resize, crop)
            images.append(image)
            positions.append(memory['sensor_angles'])
    return np.asarray(positions), np.asarray(images)


def load_data_8_dim(file_name, shape, channels, convert, resize, crop):
    images = []
    positions = []
    print 'loading training data...'
    with gzip.open(file_name, 'rb') as memory_file:
        memories = cPickle.load(memory_file)
        print 'converting training data...'
        previous_position = None
        for i, memory in enumerate(memories):
            if previous_position is not None:
                image = load_image(memory)
                image = preprocess_image(image, shape, channels, convert, resize, crop)
                images.append(image)
                current_position = memory['sensor_angles']
                delta_pos = [current_position[i] - previous_position[i] for i in range(4)]
                positions.append(previous_position + delta_pos)
            previous_position = memory['sensor_angles']
    return np.asarray(positions), np.asarray(images)


def load_data_flow_experiments(file_name, shape=(320, 240), channels=1, convert=False, resize=False, crop=False):
    images = []
    positions = []
    print 'loading training data...'
    with gzip.open(file_name, 'rb') as memory_file:
        memories = cPickle.load(memory_file)
        print 'converting training data...'
        for i in range(len(memories) - 1):
            image = load_image(memories[i])
            image = preprocess_image(image, shape, channels, convert, resize, crop)
            images.append(image)
            current_position = memories[i]['sensor_angles']
            next_position = memories[i + 1]['sensor_angles']
            delta_pos = [next_position[j] - current_position[j] for j in range(4)]
            positions.append(current_position + delta_pos)
    return np.asarray(positions), np.asarray(images)


def load_state_memory(file_name):
    with gzip.open(file_name, 'rb') as memory_file:
        state_memory = cPickle.load(memory_file)
        return state_memory


def preprocess_image(image, shape, channels, convert, resize, crop):
    if convert:
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if resize:
        image = cv2.resize(image, shape)
    if crop:
        image = crop_image(image, shape, crop)
    return image


def flow_intensity(flow):
    assert flow.shape[-1] == 2
    return np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)


def threshold_flows(positions, images, threshold):
    indices = [i for i, image in enumerate(images) if flow_intensity(image).max() > threshold]
    return positions[indices], images[indices]


def crop_image(image, target_shape, method='center'):
    shape_diff = (image.shape[0] - target_shape[0], image.shape[1] - target_shape[1])
    offsets = (shape_diff[0] / 2, shape_diff[1] / 2)
    if 'top' in method:
        distance = int(method[3:])
        return image[distance: target_shape[0] + distance, offsets[1]: offsets[1] + target_shape[1]]
    elif 'bottom' in method:
        distance = int(method[6:])
        return image[shape_diff[0] - distance: -distance, offsets[1]: offsets[1] + target_shape[1]]
    else:
        return image[offsets[0]: offsets[0] + target_shape[0], offsets[1]: offsets[1] + target_shape[1]]


def merge(file_names, new_name):
    memories = []
    for file_name in file_names:
        with gzip.open(file_name, 'rb') as memory_file:
            memories += cPickle.load(memory_file)

    with gzip.open(new_name, 'wb') as new_file:
        cPickle.dump(memories, new_file, protocol=cPickle.HIGHEST_PROTOCOL)


def cut(file_name, from_idx, to_idx):
    cut_memories = []
    with gzip.open(file_name, 'rb') as memory_file:
        memories = cPickle.load(memory_file)
        for i, memory in enumerate(memories):
            if from_idx <= i <= to_idx:
                cut_memories.append(memory)

    new_name = file_name.replace('.pkl', '_{}to{}.pkl'.format(from_idx, min(to_idx, len(memories) - 1)))
    with gzip.open(new_name, 'wb') as new_file:
        cPickle.dump(cut_memories, new_file, protocol=cPickle.HIGHEST_PROTOCOL)


def natural_sort(l):

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(l, key=alphanum_key)


if __name__ == '__main__':
    cut('data/images/grey400_bw.pkl', 0, 300)
