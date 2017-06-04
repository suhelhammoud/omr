import cv2
import numpy as np
from matplotlib import pyplot as plt
from omr_utils import *
from OmrExceptions import *

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def otsu_id_filter(img, kernel=1):
    blur = cv2.medianBlur(img, kernel, 0)  # TODO adjust the kernel
    # blur = img
    _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def get_x_splits(start, end):
    # assert end > start
    delta = (end - start) / 10
    return [start + int(i * delta + delta / 2) for i in range(10)]


def filter_y_splits(downs, ups):
    ends = downs[1:11]
    starts = ups[0:10]
    return [int((start + end) / 2) for start, end in zip(starts, ends)]

    # return ends, starts


def count_id_blob(img, x, y):
    delta_y = 3
    delta_x = 4
    norm = 4 * delta_x * delta_y * 255
    return np.sum(img[y - delta_y: y + delta_y, x - delta_x: x + delta_x]) / norm


def calc_id_number(img, x_splits, y_splits):
    # id number sample = 7036093052
    result = 0
    for idx, x in enumerate(x_splits):
        idx = 10 - idx
        digit = 0
        m_count = 0
        for idy, y in enumerate(y_splits):
            m_count = 0
            count = count_id_blob(img, x, y)
            if count < .15:
                digit = idy
                m_count += 1
                if m_count > 1:
                    logger.error('second value: %s for digit %s ', digit, idx)
        if m_count > 1:
            raise IDError('Mutliple values for Digit %s = %s' % (idx))
        result = result * 10 + digit
    return result


def id_box_border(img, debug=False):
    height, width = img.shape
    img_x = otsu_id_filter(img, kernel=1)
    sum_x = np.sum(img_x, axis=0) / height

    x_downs, x_ups = get_crossing_downs_ups(sum_x, 240)
    if not len(x_downs) == 1 and not len(x_ups) == 1:
        logger.error("downs: %s, ups: %s", x_downs, x_ups)
        raise IDError("Could not locate the vertical borders of ID box")

    x_shift_correction = 3
    x_splits = get_x_splits(x_downs[0] + x_shift_correction, x_ups[0])

    img_y = otsu_id_filter(img, kernel=5)
    sum_y = np.sum(img_y, axis=1) / width
    y_starts, y_ends = get_crossing_downs_ups(sum_y, 240, spacing=0)
    if not len(y_starts) > 0 and not len(y_ends) > 0:
        logger.error("y_starts: %s, y_ends: %s", y_starts, y_ends)
        raise IDError("Could not locate the horizontal borders of ID box")

    # slicing sum_y : [y_starts[0]: y_ends[0]]
    m_avg = np.average(sum_y[50: 285]) - 10
    logger.debug("moving avg: %s", m_avg)
    y_downs, y_ups = get_crossing_downs_ups(sum_y, m_avg, spacing=0)
    logger.debug("y_donws: %s, y_ups: %s", len(y_downs), len(y_ups))

    y_splits = filter_y_splits(y_downs, y_ups)

    # id_blobs = [get_id_blob(img_y, x, y) for y in y_splits for x in x_splits]
    #
    # for x in range(10):
    #     for y in range(10):
    #         id_blob = get_id_blob(img_y, y_splits[y], x_splits[x])
    #         index = 10* x +y
    #         plt.subplot(10, 10, index + 1)
    #         plt.imshow(id_blob, 'gray')
    #         plt.title('(%s,%s)' % (x, y))
    #         plt.show()
    # plt.show()

    if len(y_splits) < 10:
        raise IDError('Not enough y_split markers: %s' % len(y_splits))

    id_number = calc_id_number(img_y, x_splits, y_splits)
    logger.debug('id number = %s', id_number)

    if debug:
        vis_x = img_x.copy()
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

        for x in x_splits:
            for y in y_splits:
                count = count_id_blob(img_y, x, y)
                # print('%2.4s' % str(count), '\t', end='')
                # cv2.circle(vis_y, (x, y), 3, [170, 170, 170], 3)
                # print('')

        # for x in y_ups:
        #     plt.axvline(x=x)  # draw vertical lines in chart on xs
        #     cv2.line(vis_y, (0, x), (width, x), (255, 255, 255), 1)

        plt.show()

    return id_number


def process_id(id_marker, img):
    height, width = img.shape

    id_number = id_box_border(img, debug=False)
    print(id_number)
    assert id_number == 7036093052


if __name__ == '__main__':
    print("process name")
