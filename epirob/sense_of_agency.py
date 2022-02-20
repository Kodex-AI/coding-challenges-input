# author: Claus Lang
# email: claus.lang@bccn-berlin.de
import os
import sys
import keras
import cv2
import image_warp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from util import load_data, load_data_flow_experiments
from scipy.stats import ttest_ind


def prediction_error(image, prediction):
    return np.mean(np.abs(image - prediction))


def compare_errors(file_names, model_file_name, create_plot=False):
    num_trials = len(file_names)
    assert num_trials == 2
    model = keras.models.load_model(model_file_name)
    shape = model.layers[-1].output_shape[1:-1]

    positions, images = [], []
    errors = []
    predictions = []
    originals_a, originals_b = [], []
    error_maps_a, error_maps_b = [], []

    for i, file_name in enumerate(file_names):
        positions_tmp, images_tmp = load_data(8, file_name, shape, 1, convert=True, resize=True)
        _, originals = load_data(8, file_name, shape, 3, convert=True, resize=True)
        if i == 0:
            originals_a = originals
        elif i == 1:
            originals_b = originals
        positions.append(positions_tmp)
        images.append(images_tmp)
        errors.append([])

    for i in range(max([len(p) for p in positions])):
        display_image = None
        display_prediction = None
        display_error = None
        for trial_id in range(num_trials):
            if i < len(positions[trial_id]):
                position = positions[trial_id][i].reshape((1, positions[trial_id].shape[1]))
                prediction = model.predict(position).squeeze()
                image = images[trial_id][i].squeeze()
                errors[trial_id].append(prediction_error(image, prediction) / 255.)
                error_map = np.abs(image - prediction)
                mask = error_map > 70
                error_map = error_map * mask

                if trial_id == 0:
                    predictions.append(prediction)
                    error_maps_a.append(error_map)
                else:
                    error_maps_b.append(error_map)

                if display_image is None:
                    display_image = image
                    display_prediction = prediction
                    display_error = error_map
                else:
                    display_image = np.hstack((display_image, image))
                    display_prediction = np.hstack((display_prediction, prediction))
                    display_error = np.hstack((display_error, error_map))

    if create_plot:
        create_paper_plot(predictions, originals_a, originals_b, error_maps_a, error_maps_b)

    t, p = ttest_ind(errors[0], errors[1])
    print('trial', file_names[0].split('_')[1].replace('a', ''))
    print('mean error a:', np.mean(errors[0]))
    print('std a:', np.std(errors[0]))
    print('mean error b:', np.mean(errors[1]))
    print('std b:', np.std(errors[1]))
    print('p value:', '{:0.2e}'.format(p))
    print()


def create_paper_plot(predictions, originals_a, originals_b, error_maps_a, error_maps_b, start=24, step=1, length=7):

    def imshow(axis, img, cmap='gray'):
        img = img.squeeze()
        axis.imshow(img, cmap=cmap)
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        plt.setp(axis.spines.values(), visible=False)

    predictions = predictions[start:start + length * step:step]
    originals_a = originals_a[start:start + length * step:step]
    originals_b = originals_b[start:start + length * step:step]
    error_maps_a = error_maps_a[start:start + length * step:step]
    error_maps_b = error_maps_b[start:start + length * step:step]

    fig = plt.figure(figsize=(length * 3, 8.5 * 3))
    gs = gridspec.GridSpec(2, 1)
    gs.update(wspace=0.1)
    gs.update(hspace=0.1)

    gs1 = gridspec.GridSpecFromSubplotSpec(4, length, subplot_spec=gs[0], hspace=0.01, wspace=0.02)
    gs2 = gridspec.GridSpecFromSubplotSpec(4, length, subplot_spec=gs[1], hspace=0.01, wspace=0.02)

    font_size = 24
    for i in range(length):
        ax = plt.subplot(gs1[0, i])
        imshow(ax, originals_a[i])
        if i == 0:
            ax.set_ylabel(r'|$Img_t$|', fontsize=font_size)

        ax = plt.subplot(gs1[1, i])
        imshow(ax, cv2.cvtColor(originals_a[i], cv2.COLOR_BGR2GRAY))
        if i == 0:
            ax.set_ylabel(r'|$Img_t$|', fontsize=font_size)

        ax = plt.subplot(gs1[2, i])
        imshow(ax, predictions[i])
        if i == 0:
            ax.set_ylabel(r'|$Img^*_t$|', fontsize=font_size)

        ax = plt.subplot(gs1[3, i])
        imshow(ax, 1 - error_maps_a[i])
        if i == 0:
            ax.set_ylabel(r'$|Img^*_t - Img_t|$', fontsize=font_size)

        ax = plt.subplot(gs2[0, i])
        imshow(ax, originals_b[i])
        if i == 0:
            ax.set_ylabel(r'|$Img_t$|', fontsize=font_size)

        ax = plt.subplot(gs2[1, i])
        imshow(ax, cv2.cvtColor(originals_b[i], cv2.COLOR_BGR2GRAY))
        if i == 0:
            ax.set_ylabel(r'|$Img_t$|', fontsize=font_size)

        ax = plt.subplot(gs2[2, i])
        imshow(ax, predictions[i])
        if i == 0:
            ax.set_ylabel(r'|$Img^*_t$|', fontsize=font_size)

        ax = plt.subplot(gs2[3, i])
        imshow(ax, 1 - error_maps_b[i])
        if i == 0:
            ax.set_ylabel(r'$|Img^*_t - Img_t|$', fontsize=font_size)

    fig.text(0.45, 0.91, 'Condition A', va='center', fontsize=34)
    fig.text(0.45, 0.49, 'Condition B', va='center', fontsize=34)
    plt.savefig('output/images/soa_results_poster.png', bbox_inches='tight', pad_inches=0)


def warp_error_vector(model, trial, agency_dir, condition, debug=False, scale=False, verbose=False):
    shape = model.layers[-1].output_shape[1:-1]

    agency_file_names = os.listdir(agency_dir)
    file_names = [file_name for file_name in agency_file_names if '_{}{}_'.format(trial, condition) in file_name]
    file_name = file_names[0] if len(file_names) == 1 else [name for name in file_names if 'to' in name][0]

    positions, images = load_data_flow_experiments(agency_dir + file_name, shape, crop='center')

    images = [(image / 255.).astype(np.float32) for image in images]
    if model.layers[0].input_shape[-1] == 4:
        positions = [[position[np.newaxis, :4], position[np.newaxis, 4:]] for position in positions]
        flows = [model.predict(position).squeeze() for position in positions]
    else:
        flows = [model.predict(position.reshape((1, 8))).squeeze() for position in positions]
    errors = []

    warpeds = []
    for i in range(len(positions) - 1):

        warped = image_warp.forward_warp_interpolating(images[i], flows[i])
        warpeds.append(warped)

        error = prediction_error(images[i + 1], warped)
        errors.append(error)

        if debug:
            show_imgs([
                (0.5 * images[i] + 0.5 * images[i + 1], 'image overlay'),
                # (flow_to_color(flows[i], img_format='bgr'), 'OF({})'.format(i))
                # (np.sqrt(flows[i][:, :, 0] ** 2 + flows[i][:, :, 1] ** 2), '|OF({})|'.format(i))
                # (flows[i][:,:,0], 'flow[0]'),
                # (flows[i][:,:,1], 'flow[1]'),
                (warped, 'Img*({})'.format(i + 1)),
                (images[i + 1], 'Img({})'.format(i + 1)),
                (np.abs(images[i + 1] - warped), 'prediction error. mean:\n{}'.format(np.mean(error)))
            ])

    if verbose:
        return errors, {'images': images, 'warpeds': warpeds, 'flows': flows}
    else:
        return errors


def show_imgs(imgs_with_titles):
    plt.figure(figsize=(20, 5))
    num_plots = len(imgs_with_titles)
    for i, img_and_title in enumerate(imgs_with_titles):
        plt.subplot(1, num_plots, i + 1)
        img_show(img_and_title[0], img_and_title[1])
        if 'flow' in img_and_title[1]:
            plt.colorbar()
    plt.show()


def img_show(img, title=''):
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img, interpolation='none')
    else:
        # plt.imshow(img, cmap='gray', interpolation='none')
        plt.imshow(img, interpolation='none')
    plt.title(title)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])


def one_warp_trial(model, trial, agency_dir='data/images/agency/', debug=False, scale=False):

    prediction_errors_b = warp_error_vector(model, trial, agency_dir, 'b', debug, scale)
    prediction_errors_a = warp_error_vector(model, trial, agency_dir, 'a', debug, scale)

    t, p = ttest_ind(prediction_errors_a, prediction_errors_b)

    print('trial', trial)
    print('mean error a:', np.mean(prediction_errors_a))
    print('std a:', np.std(prediction_errors_a))
    print('mean error b:', np.mean(prediction_errors_b))
    print('std b:', np.std(prediction_errors_b))
    print('p value:', '{:0.2e}'.format(p))
    print()


def main():

    compare_errors([
        'data/images/agency/agency_4a_original.pkl',
        'data/images/agency/agency_4b_original.pkl'],
        'data/models/Feb19_blue500_8dim_40-79.30.h5',
        create_plot=True
    )
    return

    if len(sys.argv) > 1:
        model = sys.argv[1]
    #     trial = int(sys.argv[2])
    else:
        model = 'nao_flow_04'
    #     trial = 1

    print 'model', model, '   trial', trial

    model_file_name = 'data/models/' + [file_name for file_name in os.listdir('data/models/') if model in file_name][0]
    model = keras.models.load_model(model_file_name)
    one_warp_trial(model, trial, debug=False, scale=False, agency_dir='data/images/agency/')

    # model_file_name = 'data/models/Feb19_blue500_8dim_40-79.30.h5'
    # file_names = ['data/images/agency_new/agency_23a_original.pkl', 'data/images/agency_new/agency_23b_original.pkl']
    # compare_errors(file_names, model_file_name, create_plot=True)

    # warp_error_vector(model, 400, 'data/images/', '', debug=True, scale=False)


if __name__ == '__main__':
    main()
