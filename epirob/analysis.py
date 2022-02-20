# author: Claus Lang
# email: claus.lang@bccn-berlin.de
import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
from util import load_data, load_state_memory, threshold_flows
from images import show_img, flow_to_color
from image_warp import forward_warp_interpolating
from sense_of_agency import prediction_error, show_imgs


class Analysis:

    def __init__(self, model_file_name, positions, images, pixels, channels):
        self.pixels = pixels
        self.channels = channels
        self.model = keras.models.load_model(model_file_name)
        self.positions = positions
        self.images = images

    def predict_random(self, num_examples=8):
        shape = self.images[0].shape[:2]
        for i in range(num_examples):
            index = np.random.randint(len(self.positions))
            # index = i + 180

            position = self.positions[index].reshape((1, self.positions.shape[1]))
            image = self.images[index].reshape(shape + (self.channels,))
            prediction = self.model.predict(position).reshape(shape + (self.channels,))

            plt.subplot(2, num_examples, i + 1)
            if i == 0:
                plt.ylabel('originals', rotation=0, labelpad=30, verticalalignment='center')
            if self.channels == 1:
                self.imshow(image.reshape(shape))
            elif self.channels == 2:
                self.imshow(flow_to_color(image), cmap=None)
            else:
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            plt.subplot(2, num_examples, num_examples + i + 1)
            if i == 0:
                plt.ylabel('predictions', rotation=0, labelpad=30, verticalalignment='center')
            if self.channels == 1:
                self.imshow(prediction.reshape(shape))
            elif self.channels == 2:
                self.imshow(flow_to_color(prediction), cmap=None)
            else:
                plt.imshow(cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB))

        plt.suptitle('prediction quality on previously seen training data', fontsize=16)
        plt.show()

    def predict_sequence(self):
        num_frames = len(self.positions)
        input_dim = self.positions.shape[1]
        for i in range(num_frames):
            image = self.images[i].squeeze()
            image = image / 255.0
            position = self.positions[i].reshape((1, input_dim))
            prediction = self.model.predict(position).squeeze()
            prediction = np.minimum(prediction, 255.0) / 255.0
            display = np.hstack((image, prediction))
            show_img(display)

    def vary_variables(self, index, num_examples=7):
        position = self.positions[index]
        original = self.images[index].squeeze()

        for dimension, value in enumerate(position):
            for i in range(num_examples):
                varied_position = position.copy()
                varied_position[dimension] += -0.6 + 0.2 * (i % num_examples)
                prediction = self.model.predict(varied_position.reshape((1, 4)))
                plt.subplot(5, num_examples, dimension * num_examples + i + 1)
                self.imshow(prediction)
                if i == 0:
                    plt.ylabel('varying\nlatent\nvariable {}'.format(dimension), rotation=0, labelpad=30,
                               verticalalignment='center')

        plt.subplot(5, num_examples, 4 * num_examples + num_examples / 2 + 1)
        self.imshow(original)
        plt.xlabel('original')
        plt.show()

    def visualize_validation_4dim(self, validation_file, num=10, start_stop=None):
        positions, images = load_data(4, validation_file, self.pixels, self.channels, convert=True)

        if start_stop:
            indices = np.linspace(start_stop[0], start_stop[1], num=num).astype(int)
        else:
            indices = np.linspace(0, len(positions), num=num, endpoint=False).astype(int)

        plt.figure(figsize=(15, 4))
        plt.suptitle('varying a single joint dimension - originals not part of training data', fontsize=18)

        for i, index in enumerate(indices):
            image = images[index]
            plt.subplot(2, num, i + 1)
            self.imshow(image)
            if i == 0:
                plt.ylabel('originals', rotation=0, labelpad=30, verticalalignment='center')

        for i, index in enumerate(indices):
            position = positions[index]
            prediction = self.model.predict(position.reshape((1, self.positions.shape[1])))
            plt.subplot(2, num, num + i + 1)
            self.imshow(prediction)
            if i == 0:
                plt.ylabel('predictions', rotation=0, labelpad=30, verticalalignment='center')

        plt.show()

    def visualize_validation_8dim(self, validation_file, num=10, start_stop=None):
        positions, images = load_data(8, validation_file, self.pixels, self.channels, convert=False)

        if start_stop:
            indices = np.linspace(start_stop[0], start_stop[1], num=num).astype(int)
        else:
            indices = np.linspace(0, len(positions), num=num, endpoint=False).astype(int)

        plt.figure(figsize=(15, 4))
        plt.suptitle('varying a single joint dimension - originals not part of training data', fontsize=18)

        for i, index in enumerate(indices):
            image = images[index]
            plt.subplot(3, num, i + 1)
            self.imshow(image)
            if i == 0:
                plt.ylabel('originals', rotation=0, labelpad=30, verticalalignment='center')

        for i, index in enumerate(indices):
            position = positions[index]
            prediction = self.model.predict(position.reshape((1, self.positions.shape[1])))
            plt.subplot(3, num, num + i + 1)
            self.imshow(prediction)
            if i == 0:
                plt.ylabel('predictions', rotation=0, labelpad=30, verticalalignment='center')

        for i, index in enumerate(indices):
            position = positions[index]
            switched_position = np.zeros_like(position)
            switched_position[4:] = position[:4] + position[4:]
            prediction = self.model.predict(switched_position.reshape((1, self.positions.shape[1])))
            plt.subplot(3, num, 2 * num + i + 1)
            self.imshow(prediction)
            if i == 0:
                plt.ylabel('switched\npredictions', rotation=0, labelpad=30, verticalalignment='center')

        plt.show()

    def show_all_close_to(self, rough_position):
        for i, position in enumerate(self.positions):
            if self.isclose(position, rough_position, tol=0.1):
                print 'example found'
                self.imshow(self.images[i])
                plt.show()

    @staticmethod
    def isclose(position_1, position_2, tol):
        for i, element_1 in enumerate(position_1):
            element_2 = position_2[i]
            if abs(element_1 - element_2) > tol:
                return False
        return True

    @staticmethod
    def imshow(img, cmap='gray'):
        plt.imshow(img.squeeze(), cmap=cmap)
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])


def compare_angles(file_name):
    state_memory = load_state_memory(file_name)
    joints = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll']
    command_angles, sensor_angles = [[], [], [], []], [[], [], [], []]
    num_joints = len(state_memory[0]['command_angles'])
    for memory in state_memory:
        for i in range(num_joints):
            command_angles[i].append(memory['command_angles'][i])
            sensor_angles[i].append(memory['sensor_angles'][i])
    for i in range(num_joints):
        plt.subplot(num_joints, 1, i + 1)
        plt.plot(command_angles[i], label='command angles')
        plt.plot(sensor_angles[i], label='sensor angles')
        plt.ylabel(joints[i])
        plt.legend(loc='best')
    plt.suptitle('joint configuration record over time: LShoulderPitch gets stuck', fontsize=16)
    plt.show()


def count_samples(file_name):
    positions, images = load_data(4, file_name, 128, 1, convert=False)
    print(file_name)
    print('number of samples:', len(positions))
    print


def analyze_input(file_name):
    positions, images = load_data(8, file_name)
    positions = np.abs(np.asarray(positions))
    print('mean position magnitude:', positions[:, :4].mean())
    print('mean deltapos magnitude:', positions[:, 4:].mean())

    for image in images:
        mag = np.sqrt(image[:,:,0] ** 2 + image[:,:,1] ** 2)
        print mag.min(), mag.mean(), mag.max()


def analyze_saved_models_loss(prefix, nofix=None, plot=False):

    def file_name_key(full_file_name):
        return int(full_file_name.split('_')[-1].split('-')[0])

    losses = []
    min_loss_file = None
    max_loss_file = None
    file_names = os.listdir('data/models/')
    file_names.sort(key=file_name_key)
    for file_name in file_names:
        if not file_name.startswith(prefix):
            continue
        if nofix is not None and nofix in file_name:
            continue
        loss = float(file_name.replace('.h5', '').split('-')[-1])
        if len(losses) == 0 or loss < min(losses):
            min_loss_file = file_name
        if len(losses) == 0 or loss > max(losses):
            max_loss_file = file_name
        losses.append(loss)
    print 'min loss file:', min_loss_file
    print 'max loss file:', max_loss_file
    if plot:
        plt.plot(losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()


def flow_validation(images, flows, warp_function, warp_method, error_function, idx=None):
    errors = np.zeros(len(images) - 1)
    for i in range(len(images) - 1):
        if idx is not None and i != idx:
            continue
        if i % 10 == 0:
            print 'evaluated {} / {} images'.format(i, len(images) - 1)
        if warp_method is None or warp_method == '':
            prediction = warp_function(images[i], flows[i])
        else:
            prediction = warp_function(images[i], flows[i], warp_method)
        errors[i] = error_function(images[i + 1], prediction)

        if idx is not None:
            return images[i], images[i + 1], flows[i], prediction

    print 'using {} {} for warping'.format(warp_function.__name__, warp_method)
    print 'avg prediction error: {}'.format(errors.mean())


def validate_flow_fm(images_file, positions_file, model_file, warp_function, warp_method, error_function):
    model = keras.models.load_model(model_file)
    shape = model.layers[-1].output_shape[1:-1]
    positions, _ = load_data(8, positions_file, convert=False, scale=False)
    _, images = load_data(4, images_file, shape=shape, convert=False, crop=True)
    images = [(image / 255.).astype(np.float32) for image in images]
    flows = model.predict(positions)
    flow_validation(images, flows, warp_function, warp_method, error_function)


def validate_flow(images_file, flows_file, warp_function, warp_method, error_function, idx=None):
    shape = (192, 256)
    _, images = load_data(8, images_file, shape=shape, crop='center')
    _, flows = load_data(8, flows_file)
    images = [(image / 255.).astype(np.float32) for image in images]
    return flow_validation(images, flows, warp_function, warp_method, error_function, idx=idx)


def main():

    images_file = 'data/images/grey400_original_0to300.pkl'
    # positions_file = 'data/images/grey400_bw_0to300.pkl'
    flows_file = 'data/images/grey400_flow_0to300.pkl'
    # model_file = 'data/models/nao_flow_04_977-0.17.h5'
    # warp_function = forward_warp_interpolating
    # warp_method = 'nearest'
    # validate_flow_fm(images_file, positions_file, model_file, warp_function, warp_method, prediction_error)
    # validate_flow(images_file, flows_file, warp_function, warp_method, prediction_error)

    methods = [
        (forward_warp_interpolating, 'nearest')
    ]

    for warp_function, warp_method in methods:
        print 'trying', warp_function.__name__, warp_method
        validate_flow(images_file, flows_file, warp_function, warp_method, prediction_error)

    # model_file_name = 'data/models/nao_flow_bw_01_593-0.38.h5'
    # positions, _ = load_data(8, 'data/images/grey400_bw.pkl', convert=False, scale=True)
    # _, images = load_data(4, 'data/images/grey400_flow_bw.pkl', convert=False)
    # positions, images = threshold_flows(positions, images, 3)
    # analysis = Analysis(model_file_name, positions, images, 128, 2)
    # for _ in range(5):
    #     analysis.predict_random(num_examples=6)
    # starts_stops = [(35, 57), (0, 27), (0, 41), (0, 51)]
    # for i in range(4):
    #     analysis.visualize_validation_8dim('data/images/vary_angle_{}_original_ready.pkl'.format(str(i)),
    #                                        start_stop=starts_stops[i])


if __name__ == '__main__':
    main()
