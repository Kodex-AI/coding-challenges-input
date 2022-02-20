# author: Claus Lang
# email: claus.lang@bccn-berlin.de
from __future__ import division
import cv2
import numpy as np
import image_warp
import matplotlib.pyplot as plt
from unittest import TestCase
from sense_of_agency import prediction_error
from util import load_data, crop_image


class TestSenseOfAgency(TestCase):

    def test_prediction_error(self):
        a = np.ones((10, 10))
        b = np.zeros((10, 10))
        c = np.eye(10)
        d = -np.eye(10)

        self.assertAlmostEqual(prediction_error(a, a), 0)
        self.assertAlmostEqual(prediction_error(a, b), 1)
        self.assertAlmostEqual(prediction_error(a, c), 90 / 100)
        self.assertAlmostEqual(prediction_error(a, d), 110 / 100)

        self.assertAlmostEqual(prediction_error(b, a), 1)
        self.assertAlmostEqual(prediction_error(b, b), 0)
        self.assertAlmostEqual(prediction_error(b, c), 10 / 100)
        self.assertAlmostEqual(prediction_error(b, d), 10 / 100)


class TestDataLoading(TestCase):

    def test_load_data_8_dim(self):
        file_name = 'data/images/blue200_original.pkl'
        positions_8, _ = load_data(8, file_name, (128, 128), 1, convert=True)
        positions_4, _ = load_data(4, file_name, (128, 128), 1, convert=True)

        for i in range(len(positions_8)):
            for j in range(4):
                self.assertAlmostEqual(positions_4[i][j], positions_8[i][j])
                self.assertAlmostEqual(positions_8[i][j] + positions_8[i][4 + j], positions_4[i + 1][j])


class TestImageCropping(TestCase):

    def test_img_size(self):
        file_name = 'data/images/object_permanence/object_permanence_1_original.pkl'
        _, images = load_data(4, file_name, ())
        target_shape = (199, 242)
        image = images[0]
        cropped_image = crop_image(image, target_shape)
        self.assertEqual(cropped_image.shape[:2], target_shape)

    def test_img_content(self):
        img = cv2.imread('data/images/test/nao_test_img_original.png', cv2.IMREAD_COLOR)
        img_cropped_ref = cv2.imread('data/images/test/nao_test_img_cropped.png', cv2.IMREAD_COLOR)
        shape = img_cropped_ref.shape
        img_cropped = crop_image(img, shape)
        np.testing.assert_array_almost_equal(img_cropped, img_cropped_ref)

    def test_top_crop(self):
        img = np.zeros((240, 320))
        img[:20, :] = 1
        target_shape = (192, 256)
        target_img = np.zeros(target_shape)
        target_img[:10, :] = 1
        img_cropped = crop_image(img, target_shape, method='top10')
        self.assertEqual(img_cropped.shape, target_shape)
        np.testing.assert_array_almost_equal(img_cropped, target_img)

    def test_bottom_crop(self):
        img = np.zeros((240, 320))
        img[-20:, :] = 1
        target_shape = (192, 256)
        target_img = np.zeros(target_shape)
        target_img[-10:, :] = 1
        img_cropped = crop_image(img, target_shape, method='bottom10')
        self.assertEqual(img_cropped.shape, target_shape)
        np.testing.assert_array_almost_equal(img_cropped, target_img)


class TestWarping(TestCase):

    img = np.ones((3, 3, 1))
    img[2, 0] = 10

    flow = np.zeros((3, 3, 2))
    flow[2, 0] = (1.7, -1.1)
    flow[0, 2] = (5, -5)

    def test_interpolation_data(self):
        old_indices_expected = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
        old_values_expected = np.array([1, 1, 1, 1, 1, 1, 10, 1, 1])
        new_indices_expected = np.array([[0, 0], [0, 1], [-5, 7], [1, 0], [1, 1], [1, 2], [0.9, 1.7], [2, 1], [2, 2]])
        old_indices, old_values, new_indices = image_warp.generate_interpolation_data(self.img, self.flow)
        np.testing.assert_array_almost_equal(old_indices, old_indices_expected)
        np.testing.assert_array_almost_equal(old_values[0], old_values_expected)
        np.testing.assert_array_almost_equal(new_indices, new_indices_expected)

    def test_warp_rounding(self):
        expected = np.zeros((3, 3))
        expected[1, 2] = 10
        warped = image_warp.forward_warp_rounding(self.img.squeeze(), self.flow, 0).squeeze()
        np.testing.assert_array_almost_equal(expected, warped)

    def test_occlusion_map(self):
        flow = np.zeros((2, 2, 2))
        flow[0, 0, 0] = 1
        range_map_expected = np.array([[0, 1], [0, 0]])
        occ_map_expected = np.array([[False, True], [False, False]])
        range_map, occ_map = image_warp.occlusion_map(flow, 0, False)
        np.testing.assert_array_almost_equal(range_map_expected, range_map)
        np.testing.assert_array_almost_equal(occ_map_expected, occ_map)


def show_images(images):
    plt.figure()
    num_ims = len(images)
    for i, image in enumerate(images):
        plt.subplot(1, num_ims, i + 1)
        plt.imshow(image.squeeze(), cmap='gray', interpolation='none')
        # plt.gca().set_xticks([])
        # plt.gca().set_yticks([])
    plt.show()
