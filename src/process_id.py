import cv2
import numpy as np
from matplotlib import pyplot as plt
from omr_utils import *
from OmrExceptions import *

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def otsu_id_filter(img, kernel = 1):
    blur = cv2.medianBlur(img, kernel, 0)  # TODO adjust the kernel
    # blur = img
    _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def get_x_splits(start, end):
    assert end > start
    delta = (end - start) / 10
    return [start + int(i * delta) for i in range(11)]

def get_y_splits(start, end):
    pass


def id_box_border(img, debug=False):
    height, width = img.shape
    img_x = otsu_id_filter(img, kernel= 1)
    sum_x = np.sum(img_x, axis=0) / height

    x_downs, x_ups = get_crossing_downs_ups(sum_x, 250)
    if not len(x_downs) == 1 and not len(x_ups) == 1:
        logger.error("downs: %s, ups: %s", x_downs, x_ups)
        raise IDError("Could not locate the vertical borders of ID box")

    x_splits = get_x_splits(x_downs[0], x_ups[0])

    img_y = otsu_id_filter(img, kernel= 7)
    sum_y = np.sum(img_y, axis=1) / width
    y_downs, y_ups = get_crossing_downs_ups(sum_y, 240, spacing=0)
    if not len(y_downs)> 0 and not len(y_ups) > 0:
        logger.error("y_downs: %s, y_ups: %s", y_downs, y_ups)
        raise IDError("Could not locate the horizontal borders of ID box")

    if debug:
        vis_x = img_x.copy()
        vis_y = img_y.copy()

        plt.subplot(221), plt.imshow(vis_x, 'gray'), plt.title('img_x')
        plt.subplot(222), plt.imshow(vis_y, 'gray'), plt.title('img_y')
        plt.subplot(223), plt.plot(sum_x, 'r'), plt.title('sum_x')
        for x in x_splits:
            plt.axvline(x=x)  # draw vertical lines in chart on xs
            cv2.line(vis_x, (x, 0), (x, height), (255, 255, 255), 1)
        plt.subplot(224), plt.plot(sum_y, 'b'), plt.title('sum_y')
        plt.show()

    return x_splits


def process_id(id_marker, img):
    height, width = img.shape


    x_splits = id_box_border(img, debug=True)


if __name__ == '__main__':
    print("process name")
