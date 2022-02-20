import math
import numpy as np
import tensorflow as tf
from scipy.interpolate import griddata


def generate_interpolation_data(im, flow):
    height, width, channels = im.shape

    height_indices = np.repeat(np.arange(height), width)[:, np.newaxis]
    width_indices = np.tile(np.arange(width), height)[:, np.newaxis]
    old_indices = np.hstack((height_indices, width_indices))
    old_values = [im[:, :, channel].flatten() for channel in range(channels)]

    new_height_indices = height_indices + flow[:, :, 1].reshape((height * width, 1))
    new_width_indices = width_indices + flow[:, :, 0].reshape((height * width, 1))
    new_indices = np.hstack((new_height_indices, new_width_indices))

    return old_indices, old_values, new_indices


def forward_warp_interpolating(im, flow, method='nearest'):
    # flow = flow * np.dstack([np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2) > 1.5] * 2)
    height, width, channels = im.shape
    old_indices, old_values, new_indices = generate_interpolation_data(im, flow)
    warped = []
    for channel in range(channels):
        warped.append(griddata(points=new_indices, values=old_values[channel], xi=old_indices, method=method,
                               fill_value=im[:, :, channel].mean())
                      .reshape(height, width))
    return np.dstack(warped)


def forward_warp_rounding(im, flow, threshold):
    # todo: vectorize
    height, width = im.shape
    warped = np.zeros_like(im)
    for i in range(height):
        for j in range(width):
            new_i = int(np.round(i + flow[i, j, 1]))
            new_j = int(np.round(j + flow[i, j, 0]))
            if np.sqrt(flow[i, j, 1] ** 2 + flow[i, j, 0] ** 2) < threshold:
                continue
            if new_i == i and new_j == j:
                continue
            if 0 <= new_i < height and 0 <= new_j < width:
                warped[new_i, new_j] += im[i, j]
    return warped


def smooth_occlusion_map(occ_map, smooth_range):
    # todo: optimize
    height, width = occ_map.shape
    smoothed_map = np.zeros_like(occ_map, dtype=bool)
    smmooth_indices = np.arange(-smooth_range, smooth_range + 1, 1)
    for i in range(height):
        for j in range(width):
            if not occ_map[i, j]:
                continue
            for a in smmooth_indices:
                for b in smmooth_indices:
                    if 0 <= i + a < height and 0 <= j + b < width:
                        smoothed_map[i + a, j + b] = True
    return smoothed_map


def occlusion_map(flow, threshold, smooth=True, smooth_range=1):
    img = np.ones(flow.shape[:-1])
    range_map = forward_warp_rounding(img, flow, threshold)
    occ_map = range_map > 0
    if smooth:
        occ_map = smooth_occlusion_map(occ_map, smooth_range)
    return range_map, occ_map


def floor(x):
    return int(math.floor(x))


def ceil(x):
    return int(math.ceil(x))
