import logging
from omr_utils import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from Configuration import OmrConfiguration as conf, Marker, Section
import math

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class V:
    top_left = "top_left"
    top_right = "top_right"
    bottom_left = "bottom_left"
    bottom_right = "bottom_right"


def get_page_vertical_sides(img):
    height, width = img.shape
    left_side = np.argmax(img, axis=1)
    right_side = width - np.argmax(img[:, ::-1], axis=1)
    assert len(left_side) == len(right_side) == height

    ynz = np.nonzero(left_side)[0]

    # rleft = nz_point_side(left_side)
    # rright= nz_point_side(right_side)
    return ynz, left_side[ynz], right_side[ynz]


def stack(nz, side):
    return np.stack((nz, side), axis=1)


def get_middle_point(x, y):
    half = int(len(x) / 2)
    return x[half], y[half]


def get_center(left_x, right_x, ynz):
    l_point = get_middle_point(left_x, ynz)
    r_point = get_middle_point(right_x, ynz)
    assert l_point[1] == r_point[1]
    return int((l_point[0] + r_point[0]) / 2), l_point[1]


def split_list(a_list):
    half = int(len(a_list) / 2)
    return a_list[:half], a_list[half:]


def process_side(x, y, center_x, side="left"):
    result = {}
    # middle_point = get_middle_point(x, y)

    x_top, x_bottom = split_list(x)
    y_top, y_bottom = split_list(y)
    # logger.debug("x_top = %s", x_top)
    # logger.debug("y_top = %s", y_top)

    # Process the upper side
    if (x_top[0] - center_x) * (x_top[-1] - center_x) > 0:
        # same side
        if side == "left":
            result[V.top_left] = (x_top[0], y_top[0])
        else:
            result[V.top_right] = (x_top[0], y_top[0])
    else:  # cross sides
        corner = get_corner(x_top, y_top)
        logger.debug("corner top = %s , side = %s", corner, side)
        if side == "left":
            result[V.top_right] = (x_top[0], y_top[0])
            result[V.top_left] = corner
        else:
            result[V.top_left] = (x_top[0], y_top[0])
            result[V.top_right] = corner

    # logger.debug("x_bottom = %s", x_bottom)
    # logger.debug("y_bottom = %s", y_bottom)

    # Process the lower side
    if (x_bottom[0] - center_x) * (x_bottom[-1] - center_x) > 0:
        # same side
        if side == "left":
            result[V.bottom_left] = (x_bottom[-1], y_bottom[-1])
        else:
            result[V.bottom_right] = (x_bottom[-1], y_bottom[-1])
    else:
        # cross sides
        corner = get_corner(x_bottom, y_bottom)
        logger.debug("corner bottom = %s , side = %s", corner, side)
        if side == "left":
            result[V.bottom_right] = (x_bottom[-1], y_bottom[-1])
            result[V.bottom_left] = corner
        else:
            result[V.bottom_left] = (x_bottom[-1], y_bottom[-1])
            result[V.bottom_right] = corner

    return result


def process_side_distance(x, y, center_x, side="left"):
    result = {}
    # middle_point = get_middle_point(x, y)

    x_top, x_bottom = split_list(x)
    y_top, y_bottom = split_list(y)
    # logger.debug("x_top = %s", x_top)
    # logger.debug("y_top = %s", y_top)

    # Process the upper side
    if (x_top[0] - center_x) * (x_top[-1] - center_x) > 0:
        a = np.array([center_x, y_top[0]])
        b = np.array([x_top[-1], y_top[-1]])
        corner = get_corner(x_top, y_top, a, b)
        # logger.debug('xxx upper_corner = %s', corner)

        # same side
        if side == "left":
            result[V.top_left] = corner
        else:
            result[V.top_right] = corner
    else:  # cross sides
        # TODO calculate both vertices using the corner method
        corner = get_corner(x_top, y_top)
        logger.debug("corner top = %s , side = %s", corner, side)
        if side == "left":
            result[V.top_right] = (x_top[0], y_top[0])
            result[V.top_left] = corner
        else:
            result[V.top_left] = (x_top[0], y_top[0])
            result[V.top_right] = corner

    # logger.debug("x_bottom = %s", x_bottom)
    # logger.debug("y_bottom = %s", y_bottom)

    # Process the lower side
    if (x_bottom[0] - center_x) * (x_bottom[-1] - center_x) > 0:
        a = np.array([center_x, y_bottom[-1]])
        b = np.array([x_bottom[0], y_bottom[0]])
        corner = get_corner(x_bottom, y_bottom, a, b)
        # logger.debug('xxx bottom_corner = %s', corner)

        # same side
        if side == "left":
            result[V.bottom_left] = corner
        else:
            result[V.bottom_right] = corner
    else:
        # cross sides
        # TODO calculate both vertices using the corner method
        corner = get_corner(x_bottom, y_bottom)
        logger.debug("corner bottom = %s , side = %s", corner, side)
        if side == "left":
            result[V.bottom_right] = (x_bottom[-1], y_bottom[-1])
            result[V.bottom_left] = corner
        else:
            result[V.bottom_left] = (x_bottom[-1], y_bottom[-1])
            result[V.bottom_right] = corner

    return result


def distance(a, b, p):
    """ segment line AB, point P, where each one is an array([x, y]) """
    if all(a == p) or all(b == p):  # TODO delete this check if it takes time
        return 0
    return norm(np.cross(b - a, a - p)) / norm(b - a)


def get_corner(x, y, a=None, b=None):
    if a is None:
        a = np.array([x[0], y[0]])
    if b is None:
        b = np.array([x[-1], y[-1]])
    # logger.debug("a = %s, b = %s", a, b)
    points = np.column_stack((x, y))
    distances = [distance(a, b, p) for p in points]
    # logger.debug("corner points = %s", points)
    # logger.debug("corner distan = %s", distances)
    mx_index = np.argmax(distances)
    return x[mx_index], y[mx_index]


def get_four_corners(img_filtered):
    # img_filtered = thresh_otsu(img)
    # plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Input')
    # plt.subplot(122), plt.imshow(img_filtered, 'gray'), plt.title('Sheet')

    y, x_left, x_right = get_page_vertical_sides(img_filtered)
    # logger.debug("y len = %s, data= %s ", len(y), y)
    # logger.debug("x_left len = %s, data= %s ", len(x_left), x_left)
    # logger.debug("x_right len = %s, data= %s ", len(x_right), x_right)

    center_x, center_y = get_center(x_left, x_right, y)
    logger.debug('center_x: %s, center_y: %s', center_x, center_y)

    logger.debug("processing left side")
    # l_result = process_side(x_left, y, center_x, side="left")
    l_result = process_side_distance(x_left, y, center_x, side="left")
    logger.debug('left_side_corners = %s', l_result)

    logger.debug("processing right side")
    # r_result = process_side(x_right, y, center_x, side="right")
    r_result = process_side_distance(x_right, y, center_x, side="right")
    logger.debug('right_side_corners = %s', r_result)

    four_points = merge_results(l_result, r_result)
    logger.debug("four points = %s", four_points)
    return four_points

    # three_points = EDGE.get_three_points(left_side, ynz)
    # logger.debug('three_points = %s', three_points)
    # lside = EDGE.get(three_points, center)
    #
    # print(lside)
    # print('nz : {nz}'.format(**locals()))
    # print('top_side : {top_side}'.format(**locals()))
    # print('bottom  : {bottom_side}'.format(**locals()))


def marker_filter(img, blur_param=(14, 1), median_param=conf.marker_filter_median_blur):
    blurred = img
    blurred = cv2.blur(img, blur_param)
    blur = cv2.medianBlur(blurred, median_param, 0)
    ret, th = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    return th


def thresh_otsu(img):
    blur = cv2.medianBlur(img, 17, 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # kernel = np.ones((30, 30), np.uint8)
    # dilate = cv2.dilate(th3, kernel, iterations=1)
    dilate = th3
    return dilate


def merge_results(left, right):
    result = {}
    result.update(left)
    result.update(right)
    result[V.top_left] = left[V.top_left]
    result[V.bottom_left] = left[V.bottom_left]
    result[V.top_right] = right[V.top_right]
    result[V.bottom_right] = right[V.bottom_right]
    return result


def transform(img, vertices, shape, show=False):
    # pts1 = np.float32([[top_left], [top_right], [bottom_left], [bottom_right]])
    # pts1 = np.float32([[71, 81], [491, 68], [35, 515], [520, 520]])
    pts1 = np.float32([vertices[V.top_left], vertices[V.top_right],
                       vertices[V.bottom_left], vertices[V.bottom_right]])

    # pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([[0, 0], [shape[0], 0], [0, shape[1]], shape])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    # dst = cv2.warpPerspective(img, M, (output width, height))
    dst = cv2.warpPerspective(img, M, tuple(shape))

    if show:
        plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Input')
        plt.subplot(122), plt.imshow(dst, 'gray'), plt.title('Output')
        plt.show()
    return dst


def add_shift(vertices, shift):
    result = {key: value for key, value in vertices.items()}

    x0, y0 = result[V.top_left]
    result[V.top_left] = (x0 + shift, y0)

    x1, y1 = result[V.bottom_left]
    result[V.bottom_left] = (x1 + shift, y1)

    return result


def slide_marker(img, y_step, windows_y):
    height, width = img.shape
    for y in range(0, height, y_step):
        yield (y, img[y:y + windows_y, 0:width])


def filter_marker_y_padding(mrkr, padding_top, padding_bottom):
    return mrkr[(mrkr > padding_top) & (mrkr < padding_bottom)]


def get_ups(a, avg_smoothed, spacing=0):
    a0 = a[:-1]
    a1 = a[1:]
    id_up = np.where((a0 < (avg_smoothed - spacing))
                     & (a1 > avg_smoothed + spacing))[0]
    return id_up


def get_down_ups(a, avg_smoothed, spacing=3):
    a0 = a[:-1]
    a1 = a[1:]
    id_up = np.where((a0 < (avg_smoothed - spacing))
                     & (a1 > avg_smoothed + spacing))[0]

    id_down = np.where((a0 > (avg_smoothed + spacing))
                       & (a1 < (avg_smoothed - spacing)))[0]
    return id_down, id_up


def get_markers(a, avg_smoothed, spacing=3):
    id_down, id_up = get_down_ups(a, avg_smoothed, spacing)
    logger.debug("id_up befor filtering = %s ", len(id_up))

    id_up = filter_marker_y_padding(id_up, conf.marker_y_padding_top,
                                    conf.marker_y_padding_down)
    logger.debug("id_up after filtering = %s ", len(id_up))

    logger.debug("id_up = %s", len(id_up))

    logger.debug("id_down befor filtering = %s ", len(id_down))

    id_down = filter_marker_y_padding(id_down, conf.marker_y_padding_top,
                                      conf.marker_y_padding_down)
    logger.debug("id_down after filtering = %s ", len(id_down))
    logger.debug("id_do = %s ", len(id_down))

    m = []
    for i, j in zip(id_down, id_up):
        if Marker.can_acept(i, j):
            m.append(Marker(i, j))
    markers = np.array(m)

    if len(markers) != 63:
        logger.debug("in_up = %s", id_up)
        logger.debug("in_down = %s", id_down)
        logger.debug(' m = %s', [(i, j, j - i) for i, j in zip(id_down, id_up)])

    return markers
    # r = zip(id_down, id_up)
    # return [i for i in r]
    # return np.array(r)
    # return np.stack((id_down, id_up), axis=1)


def avg_marker_height(markers):
    import math
    # assert len(markers) == 63
    h = [j - i for i, j in markers]
    logger.debug('h = %s', h)
    return np.average(h)


def draw_vertices(img, vertices):
    for k, v in vertices.items():
        cv2.circle(img, v, 4, [255, 255, 255], 4)


def update_markers_with_x(markers_list):
    section_markers = [conf.sec_marker.translate(0, m.y0) for m in markers_list[:]]
    # marker_top = section_marker_top.crop(sheet)
    for sec_marker, marker in zip(section_markers, markers_list):
        marker_roi = sec_marker.crop(sheet)
        x0, x1 = get_marker_x0_x1(marker_roi)
        marker.set_x0_x1(x0, x1)
    result = {}
    if len(markers_list == 63):
        for i in range(len(markers_list)):
            index = i + 1
            result[index] = markers_list[i].set_id(index)
    return result


def get_marker_x0_x1(marker_roi):
    marker_roi = marker_filter(marker_roi, blur_param=(1, 7), median_param=1)

    sum_marker = marker_roi.sum(0)
    sum_marker_avg = np.average(sum_marker)
    id_down, id_up = get_down_ups(sum_marker, sum_marker_avg)

    if len(id_down) == 0 and len(id_up) < 2 :
        logger.debug('marker id_down = %s, %s', len(id_down), id_down)
        logger.debug('marker id_up = %s, %s', len(id_up), id_up)
        plt.imshow(marker_roi, 'gray')
        plt.show()
        raise Exception(" up, down, error")

    if len(id_up) == 1:
        return id_down[0], id_up[0]
    else:

        return id_down[0], id_up[1]


def draw_markers(sheet, markers):
    height, width = sheet.shape
    for m in markers.values():
        y0, y1, shift_per_x = m.y0_y1_shift()
        x0, x1 = m.x0_x1()
        shift = int(shift_per_x * width)
        if (m.id == 49):
            print("shift for marker 49", shift)
        cv2.line(sheet, (0, y0), (width, y0 + shift), (0, 255, 255), 1)
        cv2.line(sheet, (0, y1), (width, y1 + shift), (255, 255, 255), 1)
        cv2.line(sheet, (x0, y0), (x0, y1 + shift), (255, 255, 255), 1)
        cv2.line(sheet, (x1, y0), (x1, y1 + shift), (255, 255, 255), 1)


def calibre_vertical(center_x=None, roi=None):
    height, width = roi.shape
    logger.debug('marker_calibre shape = %s', roi.shape)
    # blur = cv2.medianBlur(roi, 3, 0)
    blur = roi
    ret3, bin_roi = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    y_sum = bin_roi.sum(1) / width

    down, ups = get_down_ups(y_sum, 160, spacing=0)
    logger.debug("marker_calibre down = %s, ups = %s", down, ups)
    assert len(ups) == 1 and len(down) == 1

    new_center = (down[0] + ups[0]) / 2
    return new_center - center_x
    # plt.subplot(311), plt.imshow(bin_roi, 'gray'), plt.title('roi')
    # plt.subplot(312), plt.plot(y_sum, 'r'), plt.title('y_sum')
    # plt.subplot(313), plt.imshow(roi_calibre, 'gray'), plt.title('calibre')

    # plt.show()


def calibrate_with_marker(marker, sheet,
                          marker_shift=conf.sec_marker_shift,
                          marker_calibre=conf.marker_calibre_range):
    sec = Section.of(marker, marker_shift)
    img = sec.crop(sheet)
    # img = cv2.blur(img, (1,1))
    # ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

    x_sum = img.sum(0) / sec.height()

    id_up = get_ups(x_sum, 150) # TODO adjust the average params

    if len(id_up) == 0:
        logger.debug('get_ups: %s', id_up)
        plt.imshow(img, 'gray')
        plt.show()
    border = id_up[-1]


    print(marker.id, id_up)

    roi_calibre = img[:, marker_calibre[0]: marker_calibre[1]]
    # blur = cv2.medianBlur(roi_calibre, 3, 0)
    # ret3, roi_calibre = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    marker_new = marker.translate(-sec.x1, -sec.y1)
    diff = calibre_vertical(marker_new.y1, roi_calibre)
    m_c_center = (marker_calibre[0] + marker_calibre[1]) / 2.0
    shift_per_x = diff / m_c_center
    marker_new.set_shift_y(shift_per_x)

    logger.debug('marker diff = %s', diff)
    assert diff < 30

    x_abc_relative = [28.1, 44.5, 60.5, 75.5, 91.5]
    x = [int(i * border / 100) for i in x_abc_relative]
    y = [int(marker_new.y1 + shift_per_x * i) for i in x]
    logger.debug("x = %s, y = %s", x, y)

    for point in zip(x, y):
        cv2.circle(img, point, 3, [255, 255, 255], 3)

    draw_markers(img, {marker_new.id: marker_new})
    plt.subplot(311), plt.imshow(img, 'gray'), plt.title('marker: ' + str(marker.id))
    plt.subplot(312), plt.plot(x_sum, 'r'), plt.title('x_sum')
    plt.subplot(313), plt.imshow(roi_calibre, 'gray'), plt.title('calibre')

    plt.show()
    return shift_per_x


if __name__ == '__main__':

    # file_path = '../data/colored/6.jpg'
    file_path = '../data/in2/01.jpg'
    img = cv2.imread(file_path, 0)
    # plt.imshow(img, 'gray')
    # plt.show()
    height, width = img.shape
    logger.debug("height %s, width %s", height, width)

    img_otsu = thresh_otsu(img)
    # plt.imshow(img_otsu, 'gray'), plt.title('otsu')
    # plt.show()

    vertices = get_four_corners(img_otsu)
    # v_shifted = add_shift(vertices, conf.marker_l_shift)
    # v_shifted =  vertices
    # draw_vertices(img, vertices)

    sheet = transform(img, vertices, conf.rshape, False)
    vis = sheet.copy()

    # plt.subplot(121), plt.imshow(sheet, 'gray'), plt.title('Input')
    # plt.subplot(122), plt.imshow(sheet, 'gray'), plt.title('Sheet')
    # plt.show()
    # cv2.imwrite('../data/out/sheet.jpg', sheet)
    sheet_name = conf.sec_name.crop(sheet)
    sheet_type = conf.sec_type.crop(sheet)
    sheet_one = conf.sec_one.crop(sheet)
    sheet_marker = conf.sec_marker_column.crop(sheet)
    sheet_marker = marker_filter(sheet_marker)

    y_sum = sheet_marker.sum(1)
    # y_sum = sheet_marker.sum(1) / conf.marker_r_shift
    logger.debug('max_sum = %s', y_sum[np.argmax(y_sum)])
    # y_avg = np.average(y_sum)

    avg_smoothed = smooth(y_sum, window_len=conf.marker_smooth_window, window='flat')

    logger.debug("y_sum shape = %s ", y_sum.shape)
    logger.debug("avg_s shape = %s ", avg_smoothed.shape)

    markers_list = get_markers(y_sum, avg_smoothed, conf.marker_threshold_spacing)
    if len(markers_list) != 63:
        plt.subplot(313), plt.plot(y_sum, 'r', avg_smoothed, 'b'), plt.title(
            "error ")  # very important to debug splitting points
        plt.show()
    # logger.debug('markers: %s', markers)
    # avg_m = avg_marker_height(markers)
    # logger.debug('avg marker height = %s, %s', avg_m, math.ceil(avg_m))

    # section_marker_top = conf.top_marker.translate(0, markers[0].y0)

    markers = update_markers_with_x(markers_list)

    last_shift = 0

    for m in markers.values():
        if m.id in [15, 29, 33, 37, 41, 47, 49]:
            last_shift = calibrate_with_marker(m, sheet)
        m.set_shift_y(last_shift)

    # for i in [15, 29, 33, 37, 41, 47, 49]:
    #     diff = calibrate_with_marker(markers[i], sheet)
    #     markers[i].set_shift_y(diff)
    #     logger.debug('marker %s shift = %s', i, markers[i].shift_y)

    draw_markers(vis, markers)
    # section_markers = [conf.sec_marker.translate(0, m.y0) for m in markers[:]]
    # # marker_top = section_marker_top.crop(sheet)
    # for sec_marker, marker in zip(section_markers, markers):
    #     marker_roi = sec_marker.crop(sheet)
    #     x0, x1 = get_marker_x0_x1(marker_roi)
    #     marker.set_x0_x1(x0, x1)

    # plt.subplot(311), plt.imshow(marker_roi, 'gray'), plt.title("marker_roi")
    # plt.subplot(312), plt.plot(sum_marker, 'r'), plt.title("marker_sum")
    # plt.subplot(313), plt.plot(y_sum, 'r', avg_smoothed, 'b')  # very important to debug splitting points
    # plt.show()

    # for i in avg_smoothed:
    #     print(i)

    # for i in y_sum:
    #     print(i)

    # for yy, x_img in slide_marker(sheet_marker, conf.y_step, conf.y_window):
    #     plt.imshow(x_img, 'gray'), plt.title(str(yy))
    #     plt.show()
    #
    # for m in markers:
    #     y0, y1 = m.y0_y1()
    #     x0, x1 = m.x0_x1()
    #     cv2.line(sheet, (0, y0), (width, y0), (0, 255, 255), 1)
    #     cv2.line(sheet, (0, y1), (width, y1), (255, 255, 255), 1)
    #     cv2.line(sheet, (x0, y0), (x0, y1), (255, 255, 255), 1)
    #     cv2.line(sheet, (x1, y0), (x1, y1), (255, 255, 255), 1)

    plt.subplot(231), plt.imshow(img, 'gray'), plt.title('Input')
    plt.subplot(232), plt.imshow(vis, 'gray'), plt.title('Sheet')
    plt.subplot(233), plt.imshow(sheet_marker, 'gray'), plt.title('sheet_marker')
    plt.subplot(234), plt.imshow(sheet_name, 'gray'), plt.title('Name')
    plt.subplot(235), plt.imshow(sheet_type, 'gray'), plt.title('type')
    plt.subplot(236), plt.imshow(sheet_one, 'gray'), plt.title('one')
    plt.show()
    # border(img)
    # img_filtered = pre_filters(img)
    # findCorners(img_filtered)
    # print('done')
