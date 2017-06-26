import logging

import cv2
import numpy as np
from matplotlib import pyplot as plt

from omr_configuration import OmrConfiguration as conf, Marker, Section
from omr_exceptions import MarkersNumberError, MarkerXError, MarkerCalibrateError
from omr_utils import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def _slide_marker(img, y_step, windows_y):
    # TODO to be deleted later
    height, width = img.shape
    for y in range(0, height, y_step):
        yield (y, img[y:y + windows_y, 0:width])


def _filter_marker_y_padding(markers_y_indexes, padding_y_top, padding_y_bottom):
    """
    Filter the markers indexes for padding space in the top and bottom of answer sheet

    :param markers_y_indexes:
    :param padding_y_top:
    :param padding_y_bottom:
    :return:
    """
    return markers_y_indexes[(markers_y_indexes > padding_y_top)
                             & (markers_y_indexes < padding_y_bottom)]


def _get_markers(sum_x, avg, spacing=0):
    id_down, id_up = get_crossing_downs_ups(sum_x, avg, spacing)

    id_up = _filter_marker_y_padding(id_up, conf.marker_y_padding_top,
                                     conf.marker_y_padding_down)

    id_down = _filter_marker_y_padding(id_down, conf.marker_y_padding_top,
                                       conf.marker_y_padding_down)

    m = []
    for y1, y2 in zip(id_down, id_up):
        if Marker.can_acept(y1, y2):
            m.append(Marker(y1, y2))
    markers = np.array(m)

    if len(markers) != 63:
        logger.error("MarkersNumberError markers_up = %s, markers_down = %s",
                     len(id_up), len(id_down))
        raise MarkersNumberError(" %s != 63" % len(markers))

    return markers


def _update_marker_with_x(marker, sheet=None, debug=False):
    margin = 10
    # Add black margin
    marker_line = add_left_margin(sheet[marker.y1: marker.y2, 0: 50], margin, 0)
    sum_y = np.sum(marker_line, axis=0) / marker_line.shape[0]

    avg_fixed = np.average(sum_y)
    id_down, id_up = get_crossing_downs_ups(sum_y, avg_fixed, spacing=0)  # TODO spacing =1 causes unknown error !

    if debug:
        logger.debug('marker id : %s', marker)
        plt.subplot(311), plt.imshow(marker_line, 'gray'), plt.title('marker_line')
        plt.subplot(312), plt.plot(sum_y, 'r'), plt.title('sum_y')
        plt.axhline(y=avg_fixed)
        plt.show()

    if len(id_down) < 1 or len(id_up) < 2:
        logger.error('Marker x error of: %s', marker)
        logger.error('len id_down = %s, len id_up = %s', len(id_down), len(id_up))
        raise MarkerXError(marker)

    marker.set_x1_x2(id_down[0] - margin, id_up[1] - margin)

    return marker


def _marker_filter(img, blur_param=(14, 1), median_param=3):
    blurred = img
    blurred = cv2.blur(img, blur_param)
    blur = cv2.medianBlur(blurred, median_param, 0)
    ret, th = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    return th


def _draw_markers_lines(sheet, markers):
    height, width = sheet.shape
    for m in markers:
        y0, y1, shift_per_x = m.y1_y2_shift()
        x0, x1 = m.x1_x2()
        shift = int(shift_per_x * width)
        cv2.line(sheet, (0, y0), (width, y0 + shift), (0, 255, 255), 1)
        cv2.line(sheet, (0, y1), (width, y1 + shift), (255, 255, 255), 1)
        cv2.line(sheet, (x0, y0), (x0, y1 + shift), (255, 255, 255), 1)
        cv2.line(sheet, (x1, y0), (x1, y1 + shift), (255, 255, 255), 1)


def _calibre_vertical(roi=None, debug=False):
    height, width = roi.shape
    y_sum = roi.sum(1) / width

    downs, ups = get_crossing_downs_ups(y_sum, 160, spacing=0)

    if debug:
        plt.subplot(311), plt.imshow(roi, 'gray'), plt.title('roi')
        plt.subplot(312), plt.plot(y_sum, 'r'), plt.title('y_sum')
        plt.axvline(x=downs[0])
        plt.axvline(x=ups[0])
        plt.show()

    if len(downs) != 1 or len(ups) != 1:
        logger.error('calibrate vertical shift error downs= %s, ups= %s', downs, ups)
        raise MarkerCalibrateError('shift error downs= %s, ups= %s' % (downs, ups))

    return (downs[0] + ups[0]) / 2.0


def _calibrate_with_marker(marker, sheet,
                           marker_shift=conf.sec_marker_shift,
                           marker_calibre=conf.marker_calibre_range,
                           debug=False):
    sec = Section.of(marker, marker_shift)

    margin = 10
    img = add_left_margin(sec.crop(sheet), margin, 255)  # add white margin

    x_sum = img.sum(0) / sec.height()

    downs, ups = get_crossing_downs_ups(x_sum, 220, spacing=0)

    if len(downs) < 2 or len(ups) < 2:
        logger.error("MarkerCalibrateError downs: %s, ups: %s" % (len(downs), len(ups)))
        raise MarkerCalibrateError("downs: %s, ups: %s" % (len(downs), len(ups)))

    box_start = downs[1]
    box_end = ups[1]

    if debug:
        _draw_markers_lines(img, [marker])
        plt.subplot(311), plt.imshow(img, 'gray'), plt.title('marker: ' + str(marker.id))
        plt.subplot(312), plt.plot(x_sum, 'r'), plt.title('x_sum')
        for dn in downs:
            plt.axvline(x=dn)
        for up in ups:
            plt.axvline(x=up)
        plt.show()

    roi_calibre = img[:, marker_calibre[0]:marker_calibre[1]]

    center_x1 = marker.x1 + margin
    center_x2 = (marker_calibre[0] + marker_calibre[1]) / 2
    center_y2 = _calibre_vertical(roi_calibre, debug=False)
    # logger.debug('center_y2 = %s', center_y2)
    center_y1 = marker.center_y() - sec.y1

    shift_per_x = (center_y2 - center_y1) / (center_x2 - center_x1)
    marker.set_shift_y(shift_per_x)

    x_abc_relative = [28.1, 44.5, 60.5, 75.5, 91.5]

    if debug:
        # x = []
        # for point in zip(x, y):
        #     cv2.circle(img, point, 3, [255, 255, 255], 3)

        _draw_markers_lines(img.copy(), [marker])
        plt.subplot(311), plt.imshow(img, 'gray'), plt.title('marker: ' + str(marker.id))
        plt.subplot(312), plt.plot(x_sum, 'r'), plt.title('x_sum')
        plt.subplot(313), plt.imshow(roi_calibre, 'gray'), plt.title('calibre')

        plt.show()
    return shift_per_x


def draw_h_lines_on_markers(markers, sheet):
    height, width = sheet.shape
    for m in markers:
        y_shift = int(round(m.shift_y * (width - m.x1)))

        color = (0, 0, 0) if m.id % 2 == 0 else (255, 255, 255)
        cv2.line(sheet, (m.x1, m.center_y_int()), (width, m.center_y_int() + y_shift), color)
        cv2.line(sheet, m.top_left(), m.bottom_left(), color)
        cv2.line(sheet, m.top_right(), m.bottom_right(), color)




def adjust_markers_y_shift(markers, sheet, marker_calibre_range, debug=False):
    last_shift = 0

    for m in markers:
        if m.id in [15, 29, 33, 37, 41, 47, 49]:
            last_shift = _calibrate_with_marker(m, sheet, conf.sec_marker_shift,
                                                conf.marker_calibre_range, debug=True)
            m.set_shift_y(last_shift)
            # logger.debug('adjust marker shift: %s, for marker: %s', last_shift, m)
        m.set_shift_y(last_shift)
        # logger.debug('marker %s shift = %s', m.id, m.shift_y)
        if debug:
            # _draw_markers_lines(sheet, [m])
            pass


def process_marker_column(img, sec_marker_column, debug=False):
    # sheet_marker = _marker_filter(sheet_marker,
    #                               blur_param=(14, 1),
    #                               median_param=conf.marker_filter_median_blur)
    img_markers = sec_marker_column.crop(img)

    y_sum = img_markers.sum(1) / img_markers.shape[1]

    # logger.debug('max_sum = %s', y_sum[np.argmax(y_sum)])
    # y_avg = np.average(y_sum)

    # TODO try use use fixed avg rather than smooth if possible
    avg_smoothed = smooth(y_sum, window_len=conf.marker_smooth_window, window='flat')
    avg_fixed = np.average(y_sum)

    markers_list = _get_markers(y_sum, avg_fixed, spacing=0)

    # logger.debug('debug value = %s', debug)
    if debug:
        logger.debug('inside debug value = %s', debug)
        plt.subplot(211), plt.imshow(img_markers, 'gray'), plt.title('img_markers')
        plt.subplot(212), plt.plot(y_sum, 'b', avg_smoothed, 'r'), plt.title('y_sum')
        plt.axhline(y=avg_fixed)
        plt.show()

    if len(markers_list) != 63:
        logger.error("MarkersNumberError = %s", len(markers_list))
        raise MarkersNumberError(" %s != 63" % len(markers_list))

    for index, marker in enumerate(markers_list):
        marker.set_id(index)
        _update_marker_with_x(marker, img, debug=False)

    return markers_list
