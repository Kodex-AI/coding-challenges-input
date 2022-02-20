# author: Claus Lang
# email: claus.lang@bccn-berlin.de
import sys
import naoqi
import cv2
import util
import gzip
import cPickle
import datetime
import numpy as np
import colorsys
import matplotlib.pyplot as plt
from util import load_data


class ImageRetrieval:

    def __init__(self, ip=None, port=None, fps=15):
        self.ip = ip if ip is not None else util.IP
        self.port = port if port is not None else util.PORT
        self.cam_proxy = naoqi.ALProxy('ALVideoDevice', self.ip, self.port)

        # documentation of the following settings: http://doc.aldebaran.com/2-1/family/robots/video_robot.html
        camera_id = 0       # top camera
        resolution = 1      # 320x240
        color_space = 13    # kBGr color space
        self.camera_id = self.cam_proxy.subscribeCamera('test', camera_id, resolution, color_space, fps)
        self.cam_proxy.setParam(11, 0)  # kCameraAutoExpositionID
        self.cam_proxy.setParam(12, 0)  # kCameraAutoWhiteBalanceID
        self.cam_proxy.setParam(20, 0)  # kCameraExposureAlgorithmID

    def capture(self):
        nao_image = self.cam_proxy.getImageRemote(self.camera_id)
        return nao_image

    def snapshot(self):
        image = self.capture()
        self.show(image)

    @staticmethod
    def save(name, nao_image):
        image = ImageRetrieval.convert(nao_image)
        cv2.imwrite(name + '.png', image)

    @staticmethod
    def show(nao_image):
        image = ImageRetrieval.convert(nao_image)
        cv2.imshow('nao img', image)
        cv2.waitKey(0)

    @staticmethod
    def show_gray(nao_image):
        image = ImageRetrieval.convert(nao_image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        show_img(gray_image)

    @staticmethod
    def convert(nao_image):
        if nao_image is None:
            print 'cannot capture.'
        elif nao_image[6] is None:
            print 'no image data string.'
        else:
            # translate value to mat
            values = map(ord, list(nao_image[6]))
            width = nao_image[0]
            height = nao_image[1]
            image = np.zeros((height, width, 3), np.uint8)
            i = 0
            for y in range(0, height):
                for x in range(0, width):
                    image.itemset((y, x, 0), values[i + 0])
                    image.itemset((y, x, 1), values[i + 1])
                    image.itemset((y, x, 2), values[i + 2])
                    i += 3
            return image


class Segmentation:

    def __init__(self, background, pixels=None):
        self.pixels = pixels
        self.background = self.preprocess(background)

    def preprocess(self, image):
        new_image = image
        if self.pixels:
            new_image = cv2.resize(new_image, (self.pixels, self.pixels))
        if image.ndim == 3 and image.shape[2] == 3:
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        new_image = cv2.GaussianBlur(new_image, (21, 21), 0)
        return new_image

    def segment(self, img, threshold=50):
        new_image = self.preprocess(img)
        difference = cv2.absdiff(new_image, self.background)
        thresholded = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)[1]
        return thresholded


def convert_for_nn(file_name, pixels=128):
    start_time = datetime.datetime.now()
    print 'started at', start_time
    with gzip.open(file_name, 'rb') as memory_file:
        memories = cPickle.load(memory_file)
        print 'loaded at', datetime.datetime.now()
        for i, memory in enumerate(memories):
            image = ImageRetrieval.convert(memory['image'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (pixels, pixels))
            memory['image'] = image
            if i % 100 == 0:
                print 'converted', str(i), '/', str(len(memories)), 'images'
    mid_time = datetime.datetime.now()
    print 'converting took', mid_time - start_time
    print 'converting done at', mid_time
    print 'now saving to disc...'
    with gzip.open(file_name.replace('.pkl', '_original_ready.pkl'), 'wb') as new_file:
        cPickle.dump(memories, new_file, protocol=cPickle.HIGHEST_PROTOCOL)
    end_time = datetime.datetime.now()
    print 'saving took', end_time - mid_time
    print 'total duration', end_time - start_time
    print 'done at', end_time


def show_img(img):
    if img.shape[-1] == 2:
        # flow array
        flow_visualization = flow_to_color(img, img_format='bgr')
        cv2.imshow('flow', flow_visualization)
    else:
        cv2.imshow('img', img)
    cv2.waitKey(0)


def flow_to_color(flow, img_format='rgb', max_flow=None):
    n = 8

    flow_u = flow[:, :, 0]
    flow_v = flow[:, :, 1]

    if max_flow is not None:
        max_flow = max(max_flow, 1)
    else:
        max_flow = np.max(np.abs(flow))

    mag = np.sqrt(np.sum(np.square(flow), -1))

    angle = np.arctan2(flow_v, flow_u)

    im_h = np.mod(angle / (2 * np.pi) + 1.0, 1.0)
    im_s = np.clip(mag * n / max_flow, 0, 1)
    im_v = np.clip(n - im_s, 0, 1)
    im_hsv = np.stack([im_h, im_s, im_v], -1)
    im_rgb = convert_img_hsv_to_rgb(im_hsv)

    if img_format.lower() == 'rgb':
        return im_rgb
    elif img_format.lower() == 'bgr':
        return cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)
    else:
        raise NotImplementedError('Only supported img formats so far are RGB or BGR, but not {}'.format(img_format))


def convert_img_hsv_to_rgb(img_hsv):
    img_rgb = np.zeros_like(img_hsv)
    for i in range(img_hsv.shape[0]):
        for j in range(img_hsv.shape[1]):
            img_rgb[i, j, :] = colorsys.hsv_to_rgb(img_hsv[i, j, 0], img_hsv[i, j, 1], img_hsv[i, j, 2])
    return img_rgb


def show_sequence(file_name):
    _, images = load_data(4, file_name)
    for i, image in enumerate(images):
        print 'index', i
        show_img(image)


def show_sequences(file_names):
    all_images = []
    length = sys.maxsize
    for file_name in file_names:
        positions, images = load_data(4, file_name, 128, 1)
        all_images.append(images)
        length = min(length, len(positions))
    for i in range(length):
        display = all_images[0][i]
        for j in range(1, len(all_images)):
            display = np.hstack((display, all_images[j][i]))
        show_img(display)


def convert_to_bw_diff(file_name, pixels, threshold=50):
    print 'loading memories..'
    with gzip.open(file_name, 'rb') as memory_file:
        memories = cPickle.load(memory_file)
        background = memories[0]['image']
        segmentation = Segmentation(background, pixels=pixels)
        print 'converting...'
        for memory in memories:
            memory['image'] = segmentation.segment(memory['image'], threshold=threshold)
    print 'saving...'
    with gzip.open(file_name.replace('.pkl', '_bw.pkl'), 'wb') as new_file:
        cPickle.dump(memories, new_file, protocol=cPickle.HIGHEST_PROTOCOL)


def detect_circles(image, min_radius=20, max_radius=25, param2=None, param1=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dp = 2
    if param1 is not None and param2 is not None:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=1000, param1=param1, param2=param2)
    elif param1 is not None:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=1000, param1=param1)
    elif param2 is not None:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=1000, param2=param2)
    else:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=1000)
    if circles is None:
        return image, False
    radius = circles[-1, -1, -1]
    if radius < min_radius or radius > max_radius:
        return image, False
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    return image, True


def make_video(file_name, video_name):
    with gzip.open(file_name, 'rb') as memory_file:
        memories = cPickle.load(memory_file)
    frames = [memory['image'] for memory in memories]
    first_frame = frames[0]
    height, width, layers = first_frame.shape
    video = cv2.VideoWriter(video_name + '.gif', -1, 1, (width, height))
    for frame in frames:
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()


def flow_color_code():
    max_dist = 100
    granularity = 4
    flow_x = np.linspace(-max_dist, max_dist, 2 * granularity * max_dist + 1)[np.newaxis, :]
    flow_y = np.linspace(-max_dist, max_dist, 2 * granularity * max_dist + 1)[:, np.newaxis]
    flow_x = np.repeat(flow_x, 2 * max_dist * granularity + 1, axis=0)
    flow_y = np.repeat(flow_y, 2 * max_dist * granularity + 1, axis=1)
    flow = np.dstack((flow_x, flow_y))
    img = flow_to_color(flow)
    return img


def main():
    # show_sequence('data/images/agency/agency_8b_original.pkl') # 21 - 37
    # show_sequence('data/images/agency/agency_9b_original.pkl') # 12 - 17
    # show_sequence('data/images/agency/agency_12b_original.pkl') # 12 - 17
    # show_sequence('data/images/agency_new/agency_21b_original.pkl') # 17 - 22
    # show_sequence('data/images/agency_new/agency_24b_original.pkl') # 4 - 9

    show_sequence('data/images/object_permanence/object_permanence_1_original.pkl')

if __name__ == '__main__':
    main()
