import cv2
import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import sqrt
from images import flow_to_color, flow_color_code
from analysis import validate_flow
from sense_of_agency import prediction_error, warp_error_vector
from object_permanence import generate_crop_modes, generate_detection_params, enhance, detect_circles
from image_warp import forward_warp_interpolating
from archive.archive import forward_warp_rounding, forward_warp_sampling
from util import load_data, natural_sort, load_data_flow_experiments


def warping_visualization():
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))

    for plot in range(3):
        ax = axs[plot]
        ax.set_title(['Rounding', 'Reverse Bilinear Sampling', 'Interpolation (Nearest-Neighbor)'][plot],
                     fontsize=20, weight='bold')

        grid_xs = [0.5, 0.5, 1.5, 1.5]
        grid_ys = [0.5, 1.5, 0.5, 1.5]
        grid_values = [[1, '', '', 6], [1.14, 1.4, 2.82, 3.57], [1, 5, 1, 7]][plot]

        target_xs = [0.9, 1.1, 1.9, 1.9]
        target_ys = [0.4, 1.2, -0.1, 1.3]
        target_values = [1, 5, 3, 7]

        grid = ax.scatter(grid_xs, grid_ys, s=200, label='integer pixel grid')
        targets = ax.scatter(target_xs, target_ys, marker='x', color='green', s=200, linewidths=4,
                             label='flow targets')
        ax.legend(scatterpoints=1, loc='upper left', fontsize=18)
        # fig.legend((grid, targets), ('integer pixel grid', 'flow targets'), scatterpoints=1)

        offset = (7, 7)
        for i in range(4):
            ax.annotate(grid_values[i], (grid_xs[i], grid_ys[i]), color='blue', fontsize=20,
                        xytext=offset, textcoords='offset points', weight='bold')
            ax.annotate(target_values[i], (target_xs[i], target_ys[i]), color='green', fontsize=20,
                        xytext=offset, textcoords='offset points', weight='bold')

        linewidth=3
        ax.plot([0, 2], [0, 0], color='grey', linestyle='dotted', linewidth=linewidth)
        ax.plot([0, 2], [1, 1], color='grey', linestyle='dotted', linewidth=linewidth)
        ax.plot([0, 2], [2, 2], color='grey', linestyle='dotted', linewidth=linewidth)

        ax.plot([0, 0], [0, 2], color='grey', linestyle='dotted', linewidth=linewidth)
        ax.plot([1, 1], [0, 2], color='grey', linestyle='dotted', linewidth=linewidth)
        ax.plot([2, 2], [0, 2], color='grey', linestyle='dotted', linewidth=linewidth)

        arrows = [[[0], [3], [], [3]], [[0, 2], [0, 1, 2, 3], [2], [2, 3]], [[0, 2], [1], [], [3]]][plot]

        shorten_by = 0.08
        for start, ends in enumerate(arrows):
            for end in ends:
                x = target_xs[start]
                y = target_ys[start]
                dx = grid_xs[end] - x
                dy = grid_ys[end] - y
                arrow_len = sqrt(dx ** 2 + dy ** 2)
                factor = (arrow_len - shorten_by) / arrow_len
                ax.arrow(x, y, dx * factor, dy * factor, color='green' if plot < 2 else 'blue', linewidth=2)

        ax.set_xlim([-0.2, 2.2])
        ax.set_ylim([-0.2, 2.2])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    # plt.show()
    plt.savefig('output/images/warping_visualization.png', bbox_inches='tight')


def warping_examples():

    def img_show(ax, img, title='', convert=True, grey=False):
        img = img.astype(np.float32)
        if convert:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if grey:
            ax.imshow(img, interpolation='none', cmap='Greys')
        else:
            ax.imshow(img, interpolation='none')
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    images_file = 'data/images/grey400_original_0to300.pkl'
    flows_file = 'data/images/grey400_flow_0to300.pkl'
    methods = [
        (forward_warp_rounding, ''),
        (forward_warp_sampling, ''),
        (forward_warp_interpolating, 'nearest'),
    ]

    fig_1, axs_1 = plt.subplots(1, 3, figsize=(6, 2))
    fig_2, axs_2 = plt.subplots(1, 2, figsize=(4, 8))
    fig_3, axs_3 = plt.subplots(2, 3, figsize=(7, 4))

    for i, (warp_function, warp_method) in enumerate(methods):
        img_0, img_1, flow, prediction = validate_flow(images_file, flows_file, warp_function, warp_method,
                                                       prediction_error, idx=186)
        if i == 0:
            img_show(axs_1[0], img_0, '$Img_a$')
            img_show(axs_1[1], img_1, '$Img_b$')
            img_show(axs_1[2], 0.5 * img_0 + 0.5 * img_1, r'$0.5 \cdot Img_a + 0.5 \cdot Img_b$')
            # img_show(axs_2[0], flow_to_color(flow), r'$OF_{a\rightarrow b}$', convert=False)
            img_show(axs_2[0], np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2), r'$||OF_{a\rightarrow b}||$', convert=False, grey=True)
            # img_show(axs_2[0], np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)[:, :, np.newaxis] * flow_to_color(flow), r'$OF_{a\rightarrow b}$', convert=False)
            img_show(axs_2[1], flow_color_code(), 'flow color code', convert=False)

        img_show(axs_3[0, i], prediction, '$Img_b^*$')
        error = np.abs(img_1 - prediction)
        error = error.mean(axis=2)
        img_show(axs_3[1, i], error, '$|Img_b^* - Img_b|$', grey=True, convert=False)

    # fig_1.savefig('output/images/warping_example_imgs.png', bbox_inches='tight', transparent=True)
    fig_2.savefig('output/images/warping_example_flow_presentation.png', bbox_inches='tight', transparent=True)
    # fig_3.savefig('output/images/warping_example_presentation.png', bbox_inches='tight', transparent=True)


def exploration_example():
    start_idx = 6
    num_frames = 5

    plt.figure(figsize=(8, 3))

    def img_show(ax, img, title=''):
        img = img.astype(np.float32)
        if img.shape[-1] == 2:
            img = flow_to_color(img)
        else:
            img = cv2.cvtColor(img / 255., cv2.COLOR_BGR2RGB)
        ax.imshow(img, interpolation='none')
        ax.set_title(title, fontsize=12, weight='bold')
        ax.axis('off')

    _, images = load_data(8, 'data/images/grey_400_original_180to200.pkl')
    _, flows = load_data(4, 'data/images/grey400_flow_180to200.pkl')

    images = images[start_idx: start_idx + num_frames]
    flows = flows[start_idx + 1: start_idx + num_frames]

    gs = gridspec.GridSpec(2, 2 * num_frames)

    for i in range(num_frames):
        ax = plt.subplot(gs[0, 2 * i: 2 * i + 2])
        img_show(ax, images[i], r'$Img_{' + str(180 + start_idx + i) + '}$')

    for i in range(num_frames - 1):
        ax = plt.subplot(gs[1, 2 * i + 1: 2 * i + 3])
        img_show(ax, flows[i],
                 r'$OF_{' + r'{}\rightarrow {}'.format(180 + start_idx + i, 180 + start_idx + i + 1) + r'}$')

    # plt.show()
    plt.savefig('output/images/exploration.png', bbox_inches='tight')


def soa_example():

    file_names = [
        ('data/images/agency/agency_2a_original.pkl', 44),
        ('data/images/agency/agency_2b_original.pkl', 44),
        ('data/images/agency_new/agency_23a_original.pkl', 22),
        ('data/images/agency_new/agency_23b_original.pkl', 22)
    ]

    plt.figure(figsize=(6, 4))

    def img_show(ax, img, title=''):
        img = img.astype(np.float32)
        if img.shape[-1] == 2:
            img = flow_to_color(img)
        else:
            img = cv2.cvtColor(img / 255., cv2.COLOR_BGR2RGB)
        ax.imshow(img, interpolation='none')
        ax.set_title(title, fontsize=12, weight='bold')
        ax.axis('off')

    images = [load_data(4, file_name)[1][idx] for (file_name, idx) in file_names]

    gs = gridspec.GridSpec(2, 2)

    ax = plt.subplot(gs[0, 0])
    img_show(ax, images[0], '')

    ax = plt.subplot(gs[0, 1])
    img_show(ax, images[2], '')

    ax = plt.subplot(gs[1, 0])
    img_show(ax, images[1], '')

    ax = plt.subplot(gs[1, 1])
    img_show(ax, images[3], '')

    # plt.show()
    plt.savefig('output/images/soa_example.png', bbox_inches='tight')


def soa_table():
    errors_a, errors_b = [], []
    with open('output/results/SOA_complex_blue500.md') as f:
        for line in f:
            if 'trial' in line:
                trial = line.replace(')', '').replace('\n', '').split(' ')[-1]
            elif 'mean error a' in line:
                error_a = float(line.replace(')', '').replace('\n', '').split(' ')[-1])
                errors_a.append(error_a)
            elif 'mean error b' in line:
                error_b = float(line.replace(')', '').replace('\n', '').split(' ')[-1])
                errors_b.append(error_b)
            elif 'std a' in line:
                std_a = float(line.replace(')', '').replace('\n', '').split(' ')[-1])
            elif 'std b' in line:
                std_b = float(line.replace(')', '').replace('\n', '').split(' ')[-1])
            elif 'p value' in line:
                p = line.replace(')', '').replace('\n', '').replace("'", '').split(' ')[-1]
            else:
                print '{} & {} & {:0.2e} & {:0.2e} & {:0.2e} & {:0.2e} \\\\'.format(trial, p, error_a, std_a, error_b, std_b)
                print '\\hline'
    print '{:0.2e}'.format(np.mean(errors_a))
    print '{:0.2e}'.format(np.mean(errors_b))


def soa_results():

    trial = 12
    start = 12
    length = 5
    agency_dir = 'data/images/agency/'
    model = keras.models.load_model('data/models/nao_flow_04_977-0.17.h5')

    _, info_a = warp_error_vector(model, trial, agency_dir, 'a', verbose=True)
    _, info_b = warp_error_vector(model, trial, agency_dir, 'b', verbose=True)

    def img_show(ax, img, grey=False):
        img = img.astype(np.float32)
        if img.shape[-1] == 2:
            img = flow_to_color(img)
        elif img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if grey:
            ax.imshow(img, interpolation='none', cmap='Greys')
        else:
            ax.imshow(img, interpolation='none')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        plt.setp(ax.spines.values(), visible=False)

    fig = plt.figure(figsize=(length * 4, 6))
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0.1)

    gs1 = gridspec.GridSpecFromSubplotSpec(4, length * 2, subplot_spec=gs[0], wspace=0.015, hspace=0.01)
    gs2 = gridspec.GridSpecFromSubplotSpec(4, length * 2, subplot_spec=gs[1], wspace=0.015, hspace=0.01)

    font_size = 24

    for i in range(start, start + length):
        # ax = plt.subplot(gs1[0, i - start])
        ax = plt.subplot(gs1[0, 2 * (i - start): 2 * (i - start) + 2])
        img_show(ax, info_a['images'][i])
        if i - start == 0:
            ax.set_ylabel(r'$Img_t$', fontsize=font_size)

        # ax = plt.subplot(gs1[1, i - start])
        if 2 * (i - start) + 3 < 10:
            ax = plt.subplot(gs1[1, 2 * (i - start) + 1: 2 * (i - start) + 3])
            img_show(ax, info_a['flows'][i])
        if i - start == 0:
            ax = plt.subplot(gs1[1, 0])
            ax.set_ylabel(r'$OF^*_{t \rightarrow t+1}$', fontsize=font_size - 4)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            plt.setp(ax.spines.values(), visible=False)

        # ax = plt.subplot(gs1[2, i - start])
        ax = plt.subplot(gs1[2, 2 * (i - start): 2 * (i - start) + 2])
        if i - start > 0:
            img_show(ax, info_a['warpeds'][i - 1])
        else:
            img_show(ax, np.ones((240, 320, 3)))
            ax.set_ylabel(r'$Img^*_t$', fontsize=font_size)

        # ax = plt.subplot(gs1[3, i - start])
        ax = plt.subplot(gs1[3, 2 * (i - start): 2 * (i - start) + 2])
        if i - start > 0:
            img_show(ax, 1 - np.abs(info_a['warpeds'][i - 1] - info_a['images'][i]))
        else:
            img_show(ax, np.ones((240, 320, 3)))
            ax.set_ylabel(r'$PE_t$', fontsize=font_size)

        # ax = plt.subplot(gs2[0, i - start])
        ax = plt.subplot(gs2[0, 2 * (i - start): 2 * (i - start) + 2])
        img_show(ax, info_b['images'][i])
        if i - start == 0:
            ax.set_ylabel(r'$Img_t$', fontsize=font_size)

        # ax = plt.subplot(gs2[1, i - start])
        if 2 * (i - start) + 3 < 10:
            ax = plt.subplot(gs2[1, 2 * (i - start) + 1: 2 * (i - start) + 3])
            img_show(ax, info_b['flows'][i])
        if i - start == 0:
            ax = plt.subplot(gs2[1, 0])
            ax.set_ylabel(r'$OF^*_{t \rightarrow t+1}$', fontsize=font_size - 4)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            plt.setp(ax.spines.values(), visible=False)

        # ax = plt.subplot(gs2[2, i - start])
        ax = plt.subplot(gs2[2, 2 * (i - start): 2 * (i - start) + 2])
        if i - start > 0:
            img_show(ax, info_b['warpeds'][i - 1])
        else:
            img_show(ax, np.ones((240, 320, 3)))
            ax.set_ylabel(r'$Img^*_t$', fontsize=font_size)

        # ax = plt.subplot(gs2[3, i - start])
        ax = plt.subplot(gs2[3, 2 * (i - start): 2 * (i - start) + 2])
        if i - start > 0:
            error = np.abs(info_b['warpeds'][i - 1] - info_b['images'][i])
            error = error.mean(axis=2)
            img_show(ax, error, grey=True)
        else:
            img_show(ax, np.ones((240, 320, 3)))
            ax.set_ylabel(r'$PE_t$', fontsize=font_size)

    fig.text(0.275, 0.93, 'Condition A', va='center', fontsize=16)
    fig.text(0.685, 0.93, 'Condition B', va='center', fontsize=16)

    plt.show()
    # plt.savefig('output/images/soa_results_flow_{}_presentation.png'.format(trial), bbox_inches='tight', pad_inches=0)


def op_example():

    trial = 2
    first_frame = 2
    num_frames = 7
    step = 2

    file_names = os.listdir('data/images/object_permanence/')
    file_names = [file_name for file_name in file_names if '_{}_'.format(trial) in file_name]
    file_name = file_names[0] if len(file_names) == 1 else [name for name in file_names if 'to' in name][0]

    factor = 2
    plt.figure(figsize=(num_frames * factor, factor))

    def img_show(ax, img):
        img = img.astype(np.float32)
        if img.shape[-1] == 2:
            img = flow_to_color(img)
        else:
            img = cv2.cvtColor(img / 255., cv2.COLOR_BGR2RGB)
        ax.imshow(img, interpolation='none')
        ax.axis('off')

    _, images = load_data(4, 'data/images/object_permanence/' + file_name)
    images = images[first_frame::step]

    gs = gridspec.GridSpec(1, num_frames)
    gs.update(wspace=0.01)

    for i in range(num_frames):
        ax = plt.subplot(gs[i])
        img_show(ax, images[i])

    # plt.show()
    plt.savefig('output/images/op_example.png', bbox_inches='tight')


def op_visualization():
    fig, axs = plt.subplots(1, 4, figsize=(32, 8))

    for plot in range(4):
        ax = axs[plot]
        ax.set_title(['1-Grid', r'$OF^*_{t \rightarrow t+1}$', '$RM_{t+1}$', '$OCC_{t+1}$'][plot],
                     fontsize=37)

        grid_xs = [0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5]
        grid_ys = [0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5]
        grid_values = [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [(2, 0), (0, 0), (2, -1), (0, 0), (0, 0), (1, -2), (0, 0), (0, 0), (0, 0)],
                       [1, 1, 1, 1, 1, 1, 3, 2, 1],
                       [0, 0, 0, 0, 0, 0, 1, 1, 0]][plot]

        for i in range(9):
            ax.text(grid_xs[i], grid_ys[i], grid_values[i], fontsize=26, weight='bold',
                    horizontalalignment='center', verticalalignment='center')

        linewidth=10
        ax.plot([0, 3], [0, 0], color='grey', linestyle='dotted', linewidth=linewidth)
        ax.plot([0, 3], [1, 1], color='grey', linestyle='dotted', linewidth=linewidth)
        ax.plot([0, 3], [2, 2], color='grey', linestyle='dotted', linewidth=linewidth)
        ax.plot([0, 3], [3, 3], color='grey', linestyle='dotted', linewidth=linewidth)

        ax.plot([0, 0], [0, 3], color='grey', linestyle='dotted', linewidth=linewidth)
        ax.plot([1, 1], [0, 3], color='grey', linestyle='dotted', linewidth=linewidth)
        ax.plot([2, 2], [0, 3], color='grey', linestyle='dotted', linewidth=linewidth)
        ax.plot([3, 3], [0, 3], color='grey', linestyle='dotted', linewidth=linewidth)

        ax.set_xlim([-0.2, 3.2])
        ax.set_ylim([-0.2, 3.2])
        ax.set_aspect('equal')
        ax.axis('off')

    # plt.show()
    plt.savefig('output/images/op_visualization.png', bbox_inches='tight')


def op_results():

    trial = 17
    start = 4
    num_frames = 7
    indices = range(start, start + num_frames)

    threshold = 1.5
    data_dir = 'data/images/object_permanence/'

    model_file_name = 'data/models/nao_flow_04_977-0.17.h5'
    model = keras.models.load_model(model_file_name)
    shape = model.layers[-1].output_shape[1:-1]

    file_names = [file_name for file_name in os.listdir(data_dir) if 'original.' in file_name]
    file_names = natural_sort(file_names)
    file_name = file_names[trial - 1]

    positions, images = load_data_flow_experiments(data_dir + file_name, shape, 3, convert=False, resize=False,
                                                   crop=generate_crop_modes()[trial - 1])

    def img_show(axis, img, convert=True):
        if img.shape[-1] == 2:
            img = flow_to_color(img)
        elif convert:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axis.imshow(img, interpolation='none')
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        plt.setp(axis.spines.values(), visible=False)

    print 'enhancing v1...'
    enhanced_v1, predictions, masks = enhance(positions, images, model, threshold=threshold, update_background=False)
    print 'enhancing v2...'
    enhanced_v2, _, _ = enhance(positions, images, model, threshold=threshold, update_background=True)

    images = images[indices]
    masks = masks[indices] * 255
    enhanced_v1 = enhanced_v1[indices]
    enhanced_v2 = enhanced_v2[indices]

    print 'detecting circles...'
    detection_params = generate_detection_params()[trial - 1]
    images = [detect_circles(image, **detection_params)[0] for image in images]
    enhanced_v1 = [detect_circles(enhanced_image, **detection_params)[0] for enhanced_image in enhanced_v1]
    enhanced_v2 = [detect_circles(enhanced_image, **detection_params)[0] for enhanced_image in enhanced_v2]

    fontsize = 20
    ratio = 256. / 192. * 1.03
    inches = 2
    plt.figure(figsize=(num_frames * ratio * inches, 5 * inches))

    gs = gridspec.GridSpec(5, num_frames * 2)
    white_im = np.ones_like(images[0]) * 255

    for i in range(num_frames):
        ax = plt.subplot(gs[0, 2 * i: 2 * i + 2])
        img_show(ax, images[i])
        if i == 0:
            ax.set_ylabel('$Img_t$', fontsize=fontsize)

        ax = plt.subplot(gs[1, 2 * i + 1: 2 * i + 3])
        if i < num_frames - 1:
            img_show(ax, predictions[i + 1])
        else:
            img_show(ax, white_im)
        if i == 0:
            ax = plt.subplot(gs[1, 0])
            img_show(ax, white_im)
            ax.set_ylabel(r'$OF^*_{t-1 \rightarrow t}$', fontsize=fontsize)

        ax = plt.subplot(gs[2, 2 * i: 2 * i + 2])
        if i > 0:
            img_show(ax, masks[i] * 255, convert=False)
        else:
            img_show(ax, white_im)
        if i == 0:
            ax.set_ylabel('$OCC_t$', fontsize=fontsize)

        ax = plt.subplot(gs[3, 2 * i: 2 * i + 2])
        if i > 0:
            img_show(ax, enhanced_v1[i])
        else:
            img_show(ax, white_im)
        if i == 0:
            ax.set_ylabel('enhanced v1', fontsize=15)

        ax = plt.subplot(gs[4, 2 * i: 2 * i + 2])
        if i > 0:
            img_show(ax, enhanced_v2[i])
        else:
            img_show(ax, white_im)
        if i == 0:
            ax.set_ylabel('enhanced v2', fontsize=15)

    gs.update(hspace=0.01, wspace=0.02)
    # plt.show()
    plt.savefig('output/images/op_results_{}_presentation.png'.format(trial), bbox_inches='tight', pad_inches=0)


def of_example():
    fig, axs = plt.subplots(1, 4, figsize=(32, 8))

    for plot in range(4):
        ax = axs[plot]
        ax.set_title([r'$Img_t$', r'$Img_{t+1}$', r'$OF_{t\rightarrow t+1}$', r'$OF_{t\rightarrow t+1}$'][plot],
                     fontsize=44, weight='bold')

        grid_xs = [0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5]
        grid_ys = [0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5]
        grid_values = [(2, 1), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]

        if plot == 3:
            for i in range(9):
                ax.text(grid_xs[i], grid_ys[i], grid_values[i], fontsize=29, weight='bold',
                        horizontalalignment='center', verticalalignment='center')

        linewidth = 3
        ax.plot([0, 3], [0, 0], color='grey', linewidth=linewidth)
        ax.plot([0, 3], [1, 1], color='grey', linewidth=linewidth)
        ax.plot([0, 3], [2, 2], color='grey', linewidth=linewidth)
        ax.plot([0, 3], [3, 3], color='grey', linewidth=linewidth)

        ax.plot([0, 0], [0, 3], color='grey', linewidth=linewidth)
        ax.plot([1, 1], [0, 3], color='grey', linewidth=linewidth)
        ax.plot([2, 2], [0, 3], color='grey', linewidth=linewidth)
        ax.plot([3, 3], [0, 3], color='grey', linewidth=linewidth)

        ax.set_xlim([-0.2, 3.2])
        ax.set_ylim([-0.2, 3.2])
        ax.set_aspect('equal')
        ax.axis('off')

        if plot == 0:
            ax.fill([0, 1, 1, 0], [0, 0, 1, 1], 'black')
            ax.fill([0, 1, 1, 0], [1, 1, 2, 2], 'black')
        elif plot == 1:
            ax.fill([1, 2, 2, 1], [2, 2, 3, 3], 'black')
            ax.fill([2, 3, 3, 2], [1, 1, 2, 2], 'black')
        elif plot == 2:
            ax.arrow(0.5, 1.5, 1 - 0.1, 1 - 0.1, linewidth=2, head_width=0.15, head_length=0.15, color='black')
            ax.arrow(0.5, 0.5, 2 - 0.2, 1 - 0.1, linewidth=2, head_width=0.15, head_length=0.15, color='black')

    # plt.show()
    plt.savefig('output/images/of_example.png', bbox_inches='tight')


if __name__ == '__main__':
    warping_examples()
    # soa_results()