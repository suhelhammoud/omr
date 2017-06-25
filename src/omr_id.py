import cv2
import numpy as np
from matplotlib import pyplot as plt
from omr_utils import *
from omr_exceptions import *

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_x_splits(start, end):
    # assert end > start
    delta = (end - start) / 10
    return [start + int(i * delta + delta / 2) for i in range(10)]


def filter_y_splits(downs, ups):
    ends = downs[1:11]
    starts = ups[0:10]
    return [int((start + end) / 2) for start, end in zip(starts, ends)]


def id_blob_coverage(img, x, y, threshold=None):
    """
    Percentage of marking coverage. Ideally, if marked, then coverage = 1.0 (black),
    if not marked at all then coverage = 0.0 (white)

    :param img: roi contains roughly the id_box
    :param x: int, blob x coordinates
    :param y: int, blob y coordinates
    :param threshold: float (optional), used to return either 0.0 or 1.0 values
    :return: marking percentage, value between 1.0 (fully marked) and 0.0 (not marked).
        If threshold is applied then return either 1.0 (fully marked) or 0.0 (not marked)
    """
    delta_y = 3
    delta_x = 4
    norm = 4 * delta_x * delta_y * 255
    covarage = 1 - (np.sum(img[y - delta_y: y + delta_y, x - delta_x: x + delta_x]) / norm)  # inverse percent level
    if threshold is None:
        return covarage

    return 1 if covarage >= threshold else 0


def calc_id_number(img, x_splits, y_splits):
    # id number sample = 7036093052
    result = 0
    for idx, x in enumerate(x_splits):
        idx = 10 - idx
        digit = 0
        m_count = 0
        for idy, y in enumerate(y_splits):
            m_count = 0
            coverage = id_blob_coverage(img, x, y, .85)
            if coverage == 1:  # marked
                digit = idy
                m_count += 1
                if m_count > 1:
                    logger.error('second value: %s for digit %s ', digit, idx)
        if m_count > 1:
            raise IDError('Mutliple values for Digit %s = %s' % (idx))
        result = result * 10 + digit
    return result


def process_id(img, debug=False):
    height, width = img.shape
    img_x = otsu_filter(img, blur_kernel=1)
    sum_x = np.sum(img_x, axis=0) / height

    x_downs, x_ups = get_crossing_downs_ups(sum_x, 240)
    if not len(x_downs) == 1 and not len(x_ups) == 1:
        logger.error("downs: %s, ups: %s", x_downs, x_ups)
        raise IDError("Could not locate the vertical borders of ID box")

    x_shift_correction = 3
    x_splits = get_x_splits(x_downs[0] + x_shift_correction, x_ups[0])

    img_y = otsu_filter(img, blur_kernel=5)
    sum_y = np.sum(img_y, axis=1) / width
    y_starts, y_ends = get_crossing_downs_ups(sum_y, 240, spacing=0)
    if not len(y_starts) > 0 and not len(y_ends) > 0:
        logger.error("y_starts: %s, y_ends: %s", y_starts, y_ends)
        raise IDError("Could not locate the horizontal borders of ID box")

    # slicing sum_y : [y_starts[0]: y_ends[0]]
    m_avg = np.average(sum_y[50: 285]) - 10
    # logger.debug("id avg: %s", m_avg)
    y_downs, y_ups = get_crossing_downs_ups(sum_y, m_avg, spacing=0)
    # logger.debug("y_donws: %s, y_ups: %s", len(y_downs), len(y_ups))

    y_splits = filter_y_splits(y_downs, y_ups)

    if len(y_splits) < 10:
        raise IDError('Not enough y_split markers: %s' % len(y_splits))

    id_number = calc_id_number(img_y, x_splits, y_splits)
    logger.info('id_number = %s', id_number)

    if debug:
        vis_y = img_y.copy()

        plt.subplot(221), plt.imshow(img, 'gray'), plt.title('img_x')
        plt.subplot(222), plt.imshow(vis_y, 'gray'), plt.title('img_y')
        plt.subplot(223), plt.plot(sum_x, 'r'), plt.title('sum_x')
        for x in x_splits:
            plt.axvline(x=x)  # draw vertical lines in chart on xs
            cv2.line(vis_y, (x, 0), (x, height), (255, 255, 255), 1)
        plt.subplot(224), plt.plot(sum_y, 'b'), plt.title('sum_y')
        plt.axhline(y=m_avg)
        for y in y_splits:
            plt.axvline(x=y)  # draw vertical lines in chart on xs
            cv2.line(vis_y, (0, y), (width, y), (0, 0, 0), 1)

        plt.show()

    return id_number
