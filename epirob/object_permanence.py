# author: Claus Lang
# email: claus.lang@bccn-berlin.de
from __future__ import division
import os
import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from images import detect_circles
from util import natural_sort, load_data_flow_experiments, load_data
from scipy.stats import ttest_ind
from image_warp import occlusion_map


def enhance(positions, images, model, threshold, update_background=False):
    num_output_channels = model.layers[-1].output_shape[-1]
    if num_output_channels == 1:
        return enhance_with_mask_predictor(positions, images, model, threshold, update_background)
    elif num_output_channels == 2:
        return enhance_with_flow_predictor(positions, images, model, threshold, update_background)
    else:
        raise NotImplementedError('Enhance only implemented with 1 or 2 output channels of the model, not {}.'
                                  .format(num_output_channels))


def enhance_with_flow_predictor(positions, images, model, threshold, update_background=False):
    input_dim = positions.shape[1]
    background = images[0].squeeze()
    enhanced_images = []
    predictions = []
    masks = []
    for i, image in enumerate(images):
        position = positions[i].reshape((1, input_dim))
        flow = model.predict(position).squeeze()

        range_map, occ_map = occlusion_map(flow, threshold)
        occ_map = np.dstack((occ_map,) * 3)

        _, background_mask= occlusion_map(flow, 0.0, smooth_range=3)
        background_mask = np.dstack((background_mask,) * 3)
        not_background_mask = np.logical_not(background_mask)

        mask = occ_map
        not_mask = np.logical_not(mask)

        enhanced = background * mask + image.squeeze() * not_mask

        enhanced_images.append(enhanced)
        predictions.append(flow)
        masks.append(mask)

        if update_background:
            background = background * background_mask + image.squeeze() * not_background_mask

    return np.asarray(enhanced_images), np.asarray(predictions), np.asarray(masks)


def enhance_with_mask_predictor(positions, images, model, threshold=0.5, update_background=False):
    input_dim = positions.shape[1]
    background = images[0].squeeze()
    enhanced_images = []
    predictions = []
    masks = []
    for i, image in enumerate(images):
        position = positions[i].reshape((1, input_dim))
        mask = model.predict(position).squeeze()
        mask = scale(mask, 0, 255)
        mask = cv2.resize(mask, (320, 240))
        predictions.append(mask)
        background_mask = mask > 0.1
        mask = mask > threshold
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


def scale(values, a, b):
    scaled = values.copy()
    scaled = np.maximum(scaled, a)
    scaled *= b / scaled.max()
    return scaled


def run_trial(positions, images, model, threshold=0.5, update_background=False, detection_params=None, show=True):
    if detection_params is None:
        detection_params = {}
    enhanced_images, predictions, masks = enhance(positions, images, model, threshold=threshold,
                                                  update_background=update_background)

    detected = {'original': 0, 'enhanced': 0}
    displays = []

    for i, image in enumerate(images):
        enhanced = enhanced_images[i]
        enhanced, detected_enhanced = detect_circles(enhanced, **detection_params)
        detected['enhanced'] += detected_enhanced
        # original, detected_original = detect_circles(image, **detection_params)
        # detected['original'] += detected_original
        display = np.hstack((image, enhanced))
        displays.append(display)
        if show:
            cv2.imwrite('output/videos/op01_{}.jpg'.format(i), display)
            # cv2.imshow('img', display)
            # cv2.waitKey(0)

    percent_original = round(detected['original'] / len(images) * 100, 4)
    percent_enhanced = round(detected['enhanced'] / len(images) * 100, 4)

    print 'ball detected in original: {} / {} frames ({}%)'.format(detected['original'], len(images), percent_original)
    print 'ball detected in enhanced: {} / {} frames ({}%)'.format(detected['enhanced'], len(images), percent_enhanced)
    print
    return displays, percent_original, percent_enhanced


def create_paper_plot(positions, images, model, threshold=0.5, indices=None):

    if indices is None:
        indices = range(len(positions))

    def imshow(axis, img, cmap=None, convert=True):
        img = img.squeeze()
        if convert:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axis.imshow(img, cmap=cmap, vmin=0, vmax=255)
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        plt.setp(axis.spines.values(), visible=False)

    enhanced_images_v1, predictions, masks = enhance(positions, images, model, threshold=threshold,
                                                     update_background=False)

    enhanced_images_v2, _, _ = enhance(positions, images, model, threshold=threshold,
                                       update_background=True)

    images = images[indices]
    enhanced_images_v1 = enhanced_images_v1[indices]
    enhanced_images_v2 = enhanced_images_v2[indices]
    predictions = predictions[indices]
    masks = masks[indices] * 255
    length = len(images)

    images = [detect_circles(image, min_radius=20, max_radius=40)[0] for image in images]
    enhanced_images_v1 = [detect_circles(enhanced_image, min_radius=20, max_radius=40)[0]
                          for enhanced_image in enhanced_images_v1]
    enhanced_images_v2 = [detect_circles(enhanced_image, min_radius=20, max_radius=40)[0]
                          for enhanced_image in enhanced_images_v2]

    ratio = 320. / 240. * 1.03
    inches = 2
    plt.figure(figsize=(length * ratio * inches, 5 * inches))

    gs = gridspec.GridSpec(5, length)

    for i in range(length):
        ax = plt.subplot(gs[0, i])
        imshow(ax, images[i])
        if i == 0:
            ax.set_ylabel('original')

        ax = plt.subplot(gs[1, i])
        imshow(ax, predictions[i], cmap='gray', convert=False)
        if i == 0:
            ax.set_ylabel('prediction')

        ax = plt.subplot(gs[2, i])
        imshow(ax, masks[i], cmap='gray', convert=False)
        if i == 0:
            ax.set_ylabel('mask')

        ax = plt.subplot(gs[3, i])
        imshow(ax, enhanced_images_v1[i])
        if i == 0:
            ax.set_ylabel('enhanced v1')

        ax = plt.subplot(gs[4, i])
        imshow(ax, enhanced_images_v2[i])
        if i == 0:
            ax.set_ylabel('enhanced v2')

    gs.update(hspace=0.01, wspace=0.02)
    plt.savefig('output/object_permanence_poster.png', bbox_inches='tight', pad_inches=0)


def run_many_trials(model, data_dir, detection_params, crop_modes, threshold, update_background=False, show=False):
    shape = model.layers[-1].output_shape[1:-1]
    percentages = {'original': [], 'enhanced': []}
    # hard-coded to save time
    percentages['original'] = [47.0588, 62.5, 7.6923, 5.5556, 45.8333, 53.3333, 35.0, 20.0, 55.0, 20.0, 9.0909, 34.4828,
                               3.8462, 8.8235, 28.5714, 30.0, 31.8182, 42.8571, 20.0, 26.087]
    assert len(percentages['original']) == 20

    file_names = [file_name for file_name in os.listdir(data_dir) if 'original_' in file_name]
    file_names = natural_sort(file_names)
    for i, file_name in enumerate(file_names):
        positions, images = load_data_flow_experiments(data_dir + file_name, shape, 3, convert=False, resize=False,
                                                       crop=crop_modes[i])
        print 'trial', i + 1
        _, percent_original, percent_enhanced = run_trial(positions, images, model, show=show,
                                                          threshold=threshold,
                                                          update_background=update_background,
                                                          detection_params=detection_params[i])
        # percentages['original'].append(percent_original)
        percentages['enhanced'].append(percent_enhanced)

    t, p = ttest_ind(percentages['original'], percentages['enhanced'])

    print 'total'
    print 'original:', np.mean(percentages['original'])
    print 'enhanced:', np.mean(percentages['enhanced'])
    print 'p-value:', '{:0.2e}'.format(p)


def generate_detection_params():
    min_radii = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    max_radii = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 25, 25, 25, 25, 25, 25, 25, 25, 40, 40]
    param2s = [None, None, None, .1, .1, .1, None, .1, .1, .1, .01, .1, .1, .1, .1, .1,
               .01,
               .1, .1, .01]
    param1s = [None, None, None, 300, None, 250, None, 200, None, None, 70, None, None, None, None, None,
               200,
               400,
               150, 150]
    params = [{'min_radius': min_radii[i], 'max_radius': max_radii[i], 'param2': param2s[i], 'param1': param1s[i]}
              for i in range(len(param2s))]
    return params


def generate_crop_modes():
    return ['center', 'center', 'top3', 'top12', 'center', 'center', 'bottom10', 'center', 'center', 'bottom13',
            'top0', 'center', 'top0', 'top0', 'center', 'center', 'center', 'center', 'center', 'bottom15']


def main():
    # model_file_name = 'data/models/nao_flow_04_977-0.17.h5'
    # model = keras.models.load_model(model_file_name)
    #
    # threshold = 1.5
    #
    # run_many_trials(model, 'data/images/object_permanence/', generate_detection_params(), generate_crop_modes(),
    #                 threshold, update_background=False, show=True)

    positions, images = load_data(8, 'data/images/object_permanence/object_permanence_1_original_25to42.pkl')

    model_file_name = 'data/models/Feb23_blue200_bw_8dim_48-112.72.h5'
    model = keras.models.load_model(model_file_name)

    # create_paper_plot(positions, images, model, indices=range(3, 14, 2), threshold=5)
    run_trial(positions, images, model, threshold=5, detection_params=generate_detection_params()[0])


if __name__ == '__main__':
    main()
