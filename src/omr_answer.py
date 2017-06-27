import logging

import numpy as np
import cv2
from math import ceil
from matplotlib import pyplot as plt
from math import copysign

from omr_configuration import Section
from omr_exceptions import AnswerXBorderError, AnswerCalibrateError, AnswerMiddleError
from omr_utils import get_crossing_downs_ups

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_answers(sheet, m_dict):
    result = {}

    for num, (x, y) in m_dict.items():
        if num == 1 or num == 50:
            continue

        answer = get_answer(sheet, num, x, y + 3, debug=False)
        logger.info("answer = %s", answer)
        # answer.show(sheet)
        result[answer.num] = answer

    return result


def _calibre_vertical(roi=None, avg=160, debug=False):
    height, width = roi.shape
    y_sum = roi.sum(1) / width

    downs, ups = get_crossing_downs_ups(y_sum, avg, spacing=0)

    if debug or len(downs) + len(ups) > 2:
        plt.subplot(311), plt.imshow(roi, 'gray'), plt.title('roi')
        plt.subplot(312), plt.plot(y_sum, 'r'), plt.title('y_sum')
        # plt.axvline(x=downs[0])
        # plt.axvline(x=ups[0])
        plt.show()

    if len(downs) != 1 or len(ups) != 1:
        logger.error('calibrate vertical shift error downs= %s, ups= %s', downs, ups)
        raise AnswerCalibrateError('shift error downs= %s, ups= %s' % (downs, ups))

    return (downs[0] + ups[0]) / 2.0


def get_shift_y_per_x(img, x1, x2):
    pass


def get_answer(sheet, num, x_left, y_center, debug=False):
    y_span = 20
    x_span = 240
    x_right = x_left + x_span
    y_up = y_center - y_span
    y_bottom = y_center + y_span

    img_a = sheet[y_up: y_bottom, x_left:x_right]
    img_a_height, width_a = img_a.shape

    sum_x = np.sum(img_a, axis=0) / img_a_height

    # plt.subplot(211), plt.imshow(img_a, 'gray'), plt.title('crop_with' + str(num))
    # plt.subplot(212), plt.plot(sum_x, 'b'), plt.title('sum_x')
    # plt.show()

    downs, ups = get_crossing_downs_ups(sum_x, 245)
    if len(downs) < 1 and len(ups) < 1:
        logger.debug("len(downs):%s, len(ups): %s", len(downs), len(ups))
        raise AnswerXBorderError(" len(downs):%s, len(ups): %s" % (len(downs), len(ups)))

    a_x1 = downs[0]
    a_x2 = ups[0]
    a_width = a_x2 - a_x1

    calibre = [.263, .811]
    dx = 2

    x_calibre = [int(round(a_width * r)) + a_x1 for r in calibre]
    y_calibre = [int(round(_calibre_vertical(img_a[:, x - dx:x + dx], 150, debug=False)))
                 for x in x_calibre]

    a_shift = (y_calibre[1] - y_calibre[0]) / (x_calibre[1] - x_calibre[0])

    y1 = int(round(y_calibre[0] - (x_calibre[0] - a_x1) * a_shift))
    y2 = int(round(y_calibre[1] + (a_x2 - x_calibre[1]) * a_shift))

    result = Answer(num, x_left + a_x1, x_left + a_x2, y_up + y1, y_up + y2)
    # shc = sheet.copy()
    # result.draw_lines(shc)
    # result.show(shc, str(result.num))

    if debug:
        calibre = [.263, .811]
        ratio = [.193, .372, .55, .729, .908]

        # ratio = [0.184,0.369,0.548,0.733,0.908]

        color = (0, 0, 0) if num % 2 == 0 else (255, 255, 255)

        cv2.line(img_a, (a_x1, y1), (a_x2, y2), color, 1)
        plt.subplot(211), plt.imshow(img_a, 'gray'), plt.title(str(num))
        plt.subplot(212), plt.plot(sum_x, 'r'), plt.title(str(num))

        for x in [int(round(r * a_width)) + a_x1 for r in ratio]:
            cv2.line(img_a, (x, 0), (x, img_a_height), color, 1)

        for x in downs:
            plt.axvline(x=x)
        for x in ups:
            plt.axvline(x=x)

        plt.show()
        # y_calibre_2 = y1 + _calibre_vertical(answer_img[:, -x_shift - x_delta: -x_shift], debug=True)

    logger.debug('num:%s, y1:%s, y2:%s', num, y1, y2)
    return result


def answer_blob_coverage(img, x, y,
                         delta=(4, 3),
                         threshold=None):
    """
    Percentage of marking coverage. Ideally, if marked, then coverage = 1.0 (black),
    if not marked at all then coverage = 0.0 (white)

    :param img: roi contains roughly the id_box
    :param x: int, blob x coordinates
    :param y: int, blob y coordinates
    :param delta: tuple(int,int), blob scan diameter (dx,dy)
    :param threshold: float (optional), used to return either 0.0 or 1.0 values
    :return: marking percentage, value between 1.0 (fully marked) and 0.0 (not marked).
        If threshold is applied then return either 1.0 (fully marked) or 0.0 (not marked)
    """
    delta_y = delta[0]
    delta_x = delta(1)
    norm = 4 * delta_x * delta_y * 255
    covarage = 1 - (np.sum(img[y - delta_y: y + delta_y, x - delta_x: x + delta_x]) / norm)  # inverse percent level
    if threshold is None:
        return covarage
    elif covarage >= threshold:
        return 1
    else:
        return 0
        # return 1 if covarage >= threshold else 0


class Answer:
    X = [.193, .372, .55, .729, .908]

    # ratio = [0.184, 0.369, 0.548, 0.733, 0.908]

    # TODO check shift results shown between line/crosses through choices
    def __init__(self, num, x1, x2, y1, y2, span=20):
        self.num = int(round(num))
        self.x1 = int(round(x1))
        self.x2 = int(round(x2))
        self.y1 = int(round(y1))
        self.y2 = int(round(y2))
        self.span = int(round(span))

        width = self.x2 - self.x1
        y_shift = (y2 - y1) / (x2 - x1)
        self.choice_x = [int(round(xr * width) + x1) for xr in Answer.X]
        self.choice_y = [int(ceil(x * y_shift) + y1) for x in self.choice_x]

    def choices(self):
        return zip(self.choice_x, self.choice_y)

    def __str__(self):
        return "Answer {>> num:%s, color=%s, x1=%s, x2=%s, y1=%s, y2=%s }" \
               % (self.num, self.color_name(), self.x1, self.x2, self.y1, self.y2)

    def color_name(self):
        return ['GRAY', 'WHITE'][self.num % 2]

    def crop(self, sheet):
        return sheet[self.y1 - self.span: self.y1 + self.span, self.x1: self.x2]

    def draw_lines(self, sheet):
        d = 4
        color = (0, 0, 0) if self.num % 2 == 0 else (255, 255, 255)
        cv2.line(sheet, (self.x1, self.y1), (self.x2, self.y2), color, 1)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(sheet, str(self.num), (self.x1, self.y1), font, 1, (255, 255, 255), 2)

        for c in self.choices():
            cv2.line(sheet, (c[0] - d, c[1]), (c[0] + d, c[1]), color, 1)
            cv2.line(sheet, (c[0], c[1] - d), (c[0], c[1] + d), color, 1)

    def show(self, sheet, title=None):
        if title is None:
            title = self.num
        plt.imshow(self.crop(sheet), 'gray'), plt.title(str(title))
        plt.show()

    def mark(self, img, threshold=None):
        return [answer_blob_coverage(img, xi, yi, (4, 3), threshold=threshold)
                for xi, yi in self.choices()]

    @staticmethod
    def middle(a1, a2):
        num = (a2.num + a1.num) / 2
        if num % 2 != 0:
            logger.error('cannot find middle ans1: %s, ans2: %s', a1.num, a2.num)
            raise AnswerMiddleError('cannot find middle ans1: %s, ans2: %s' % (a1.num, a2.num))

        return Answer(num,
                      (a1.x1 + a2.x1) / 2,
                      (a1.x2 + a2.x2) / 2,
                      (a1.y1 + a2.y1) / 2,
                      (a1.y2 + a2.y2) / 2,
                      a1.height)

    @staticmethod
    def next(a1, a2):
        """

        :param a1:
        :param a2: a2 > a1
        :return:
        """
        dnum = a2.num - a1.num
        r = (1.0) / abs(dnum)
        return Answer(a2.num + copysign(1, dnum),
                      a2.x1 + (a2.x1 - a1.x1) * r,
                      a2.x2 + (a2.x2 - a1.x2) * r,
                      a2.y1 + (a2.y1 - a1.y1) * r,
                      a2.y2 + (a2.y2 - a1.y2) * r,
                      a2.span)
