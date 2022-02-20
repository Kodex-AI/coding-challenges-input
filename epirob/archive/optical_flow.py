import cv2
import numpy as np


def load_frames(first_index, directory='data/images/nao_raw/', prefix='nao_img_'):
    file_name1 = directory + prefix + str(first_index).zfill(6) + '.png'
    file_name2 = directory + prefix + str(first_index + 1).zfill(6) + '.png'
    frame1 = cv2.imread(file_name1)
    frame2 = cv2.imread(file_name2)
    return frame1, frame2


def show_optical_flow(frame1, frame2, hsv):
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('flow', bgr)
    cv2.waitKey(0)


def main(first_index):

    frame1, frame2 = load_frames(first_index)

    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    show_optical_flow(frame1, frame2, hsv)

    cv2.destroyAllWindows()


def show_image(idx, directory='data/images/nao_raw/', prefix='nao_img_'):
    file_name = directory + prefix + str(idx).zfill(6) + '.png'
    frame = cv2.imread(file_name)
    cv2.imshow('frame ' + str(idx), frame)
    cv2.waitKey(0)


if __name__ == '__main__':
    for idx in range(30, 141):
        show_image(idx)
    for idx in range(30, 141):
        main(idx)
