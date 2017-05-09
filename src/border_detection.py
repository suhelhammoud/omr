import logging

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from Configuration import OmrConfiguration as conf

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class V:
    top_left = "top_left"
    top_right = "top_right"
    bottom_left = "bottom_left"
    bottom_right = "bottom_right"


class VPoint:
    A = 0
    B = 1
    C = 2
    D = 3
    ERROR = 10000000

    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy

    def which(self, x, y):
        if x < self.cx:
            if y < self.cy:
                return VPoint.A
            else:
                return VPoint.D
        else:
            if y < self.cy:
                return VPoint.B
            else:
                return VPoint.C


def otsu(img):
    # global thresholding
    # img = cv2.GaussianBlur(img, (11, 11), 0)

    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # Otsu's thresholding
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    # blur = cv2.GaussianBlur(img, (5, 5), 0)
    blur = cv2.medianBlur(img, 5, 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plot all the images and their histograms
    images = [img, 0, th1,
              img, 0, th2,
              blur, 0, th3]
    titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
              'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
              'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
        plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    plt.show()


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


def distance(a, b, p):
    """ segment line AB, point P, where each one is an array([x, y]) """
    if all(a == p) or all(b == p):  # TODO delete this check if it takes time
        return 0
    return norm(np.cross(b - a, a - p)) / norm(b - a)


def get_corner(x, y):
    a = np.array([x[0], y[0]])
    b = np.array([x[-1], y[-1]])
    # logger.debug("a = %s, b = %s", a, b)
    points = np.column_stack((x, y))
    distances = [distance(a, b, p) for p in points]
    # logger.debug("corner points = %s", points)
    # logger.debug("corner distan = %s", distances)
    mx_index = np.argmax(distances)
    return x[mx_index], y[mx_index]


def get_four_corners(img):
    img_filtered = pre_filters(img)
    # plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Input')
    # plt.subplot(122), plt.imshow(img_filtered, 'gray'), plt.title('Sheet')
    # plt.show()
    y, x_left, x_right = get_page_vertical_sides(img_filtered)
    # logger.debug("y len = %s, data= %s ", len(y), y)
    # logger.debug("x_left len = %s, data= %s ", len(x_left), x_left)
    # logger.debug("x_right len = %s, data= %s ", len(x_right), x_right)

    center_x, center_y = get_center(x_left, x_right, y)
    logger.debug('center_x: %s, center_y: %s', center_x, center_y)

    logger.debug("processing left side")
    l_result = process_side(x_left, y, center_x, side="left")
    logger.debug('left_side_corners = %s', l_result)

    logger.debug("processing right side")
    r_result = process_side(x_right, y, center_x, side="right")
    logger.debug('right_side_corners = %s', r_result)

    four_points = merge_results(l_result, r_result)
    print(four_points)
    return four_points

    # three_points = EDGE.get_three_points(left_side, ynz)
    # logger.debug('three_points = %s', three_points)
    # lside = EDGE.get(three_points, center)
    #
    # print(lside)
    # print('nz : {nz}'.format(**locals()))
    # print('top_side : {top_side}'.format(**locals()))
    # print('bottom  : {bottom_side}'.format(**locals()))


def getSides(a):
    # inverted = cv2.bitwise_not(a)

    height, width = a.shape

    xx = np.arange(width)
    xy = np.arange(height)

    a0 = np.argmax(a, axis=0)
    a00 = np.argmax(a[::-1, :], axis=0)

    a00 = height - a00

    a1 = np.argmax(a, axis=1)
    a11 = np.argmax(a[:, ::-1], axis=1)
    a11 = width - a11

    # a0 = np.nonzero(a0)
    # a1 = np.nonzero(a1)
    return xx, a0, a00, xy, a1, a11


def marker_filter(img):
    blurred = img
    blurred = cv2.blur(img, (14, 1))
    blur = cv2.medianBlur(blurred, conf.marker_filter_blur, 0)
    ret, th = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    return th


def pre_filters(img):
    blur = cv2.medianBlur(img, 17, 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # kernel = np.ones((30, 30), np.uint8)
    # dilate = cv2.dilate(th3, kernel, iterations=1)

    dilate = th3
    return dilate


def border(img):
    # global thresholding
    # img = cv2.GaussianBlur(img, (11, 11), 0)


    # Otsu's thresholding after Median filtering
    blur = cv2.medianBlur(img, 17, 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print('ret3 ' + str(ret3))

    kernel = np.ones((30, 30), np.uint8)
    dilate = cv2.dilate(th3, kernel, iterations=1)

    dilate = th3
    # h = th3.sum(0)
    # v = th3.sum(1)

    (xx, a0, a00, xy, a1, a11) = getSides(dilate)

    # dh = np.diff(dh)
    # dv = np.diff(dv)

    # xh = np.arange(0, len(h))
    # xdh = np.arange(0, len(dh))


    plt.subplot(2, 2, 1)
    plt.imshow(img, 'gray')
    plt.title('original image'), plt.xticks([]), plt.yticks([])

    # plt.subplot(3, 2, 2)
    # plt.imshow(blur, 'gray')
    # plt.title('median blure'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 3)
    plt.imshow(th3, 'gray')
    plt.title('otsu thresholding'), plt.xticks([]), plt.yticks([])

    # plt.subplot(3, 4, 4)
    # plt.imshow(a_r, 'gray')
    # plt.title('reversed'), plt.xticks([]), plt.yticks([])



    # plt.subplot(3,4,5)
    # plt.plot(xx, a0,'r', xx, a00, 'g')
    # plt.title('a0'), plt.xticks([]), plt.yticks([])
    #
    # plt.subplot(3,4,6)
    # plt.plot(xy, a1, 'r', xy, a11, 'g')
    # plt.title('a1'), plt.xticks([]), plt.yticks([])


    plt.subplot(2, 2, 2)
    nz0 = np.nonzero(a0)[0]
    plt.plot(xx[nz0], a0[nz0], 'r', xx[nz0], a00[nz0], 'g')
    plt.title('nz scan_x'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 4)
    nz1 = np.nonzero(a1)[0]
    plt.plot(a1[nz1], xy[nz1], 'r', a11[nz1], xy[nz1], 'g')
    plt.title('nz scan_y'), plt.xticks([]), plt.yticks([])

    plt.show()


def findCorners(img):
    height, width = img.shape

    cx = width / 2
    cy = height / 2

    vertex = VPoint(cx, cy)

    print("cx = {cx}, cy = {cy}".format(**locals()))
    xx = np.arange(width)
    xy = np.arange(height)

    scan_x = np.argmax(img, axis=0)  # indexes of first white pixel
    scan_xr = height - np.argmax(img[::-1, :], axis=0)
    x_nz = np.nonzero(scan_x)[0]
    scan_x_nz = scan_x[x_nz]
    scan_xr_nz = scan_xr[x_nz]

    np.save('../data/pickles/a', scan_x_nz)
    # print(x_nz)
    # print(scan_x_nz)
    # print(scan_xr_nz)

    # start finding vertexes
    # lower line
    x_left = x_nz[0]
    y_left = scan_x_nz[0]
    if y_left > cy:
        y_left = scan_xr_nz[0]

    x_right = x_nz[-1]
    y_right = scan_x_nz[-1]
    if y_right > cy:
        y_right = scan_xr_nz[-1]

    print(vertex.which(x_left, y_left))
    print('x_left {x_left}, y_left {y_left}'.format(**locals()))
    print(vertex.which(x_right, y_right))
    print('x_right {x_right}, y_right {y_right}'.format(**locals()))

    # min values for the lower line
    ymin_index = np.argmin(scan_x_nz)
    xmin = x_nz[ymin_index]
    ymin = scan_x_nz[ymin_index]

    print(vertex.which(xmin, ymin))
    print("xmin = {xmin}, ymin = {ymin}".format(**locals()))

    # max values for the upper line
    ymax_index = np.argmax(scan_xr_nz)
    xmax = x_nz[ymax_index]
    ymax = scan_xr_nz[ymax_index]

    print(vertex.which(xmax, ymax))
    print("xmax = {xmax}, ymax = {ymax}".format(**locals()))

    print('----------------')
    scan_y = np.argmax(img, axis=1)
    scan_yr = width - np.argmax(img[:, ::-1], axis=1)
    y_nz = np.nonzero(scan_y)[0]
    scan_y_nz = np.nonzero(scan_y)
    scan_y_nz = scan_y[y_nz]
    scan_yr_nz = scan_yr[y_nz]

    yy_left = y_nz[0]
    xx_left = scan_y_nz[0]
    if xx_left > cx:
        xx_left = scan_yr_nz[0]

    yy_right = y_nz[-1]
    xx_right = scan_y_nz[-1]
    if xx_right > cx:
        xx_right = scan_yr_nz[-1]

    print(vertex.which(xx_left, yy_left))
    print('xx_left {xx_left}, yy_left {yy_left}'.format(**locals()))
    print(vertex.which(xx_right, yy_right))
    print('xx_right {xx_right}, yy_right {yy_right}'.format(**locals()))

    # min values for the lower line
    xmin_index = np.argmin(scan_x_nz)
    xmin = x_nz[ymin_index]
    ymin = scan_x_nz[ymin_index]

    print(vertex.which(xmin, ymin))
    print("xmin = {xmin}, ymin = {ymin}".format(**locals()))

    # max values for the upper line
    ymax_index = np.argmax(scan_xr_nz)
    xmax = x_nz[ymax_index]
    ymax = scan_xr_nz[ymax_index]

    print(vertex.which(xmax, ymax))
    print("xmax = {xmax}, ymax = {ymax}".format(**locals()))

    return (xx, scan_x, scan_xr, xy, scan_y, scan_yr)


def merge_results(left, right):
    result = {}
    result.update(left)
    result.update(right)
    result[V.top_left] = left[V.top_left]
    result[V.bottom_left] = left[V.bottom_left]
    result[V.top_right] = right[V.top_right]
    result[V.bottom_right] = right[V.bottom_right]
    return result


def foo(x_nz, scan_x_nz):
    y0 = scan_x_nz[0]
    y1 = scan_x_nz[-1]

    a = scan_x_nz - y0
    b = scan_x_nz - y1
    x = np.range(len(a))

    v0 = np.row_stack((x, a))
    v1 = np.row_stack((x, b))


def law_of_cosines(a, x, b):
    xa = a - x
    xc = b - x
    # calculate angle
    cosine_angle = np.dot(xa, xc) / (np.linalg.norm(xa) * np.linalg.norm(xc))

    angle = np.arccos(cosine_angle)
    return angle


    # pAngle = np.degrees(angle)


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
    result[V.top_left] = (x0 - shift, y0)

    x1, y1 = result[V.bottom_left]
    result[V.bottom_left] = (x1 - shift, y1)

    return result


def slide_marker(img, y_step, windows_y):
    height, width = img.shape
    for y in range(0, height, y_step):
        yield (y, img[y:y + windows_y, 0:width])


def smooth(x, window_len=150, window='flat'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    assert x.ndim == 1

    assert x.size > window_len

    if window_len < 3:
        return x

    assert window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    # return y
    return y[int(window_len / 2 - 1):-int(window_len / 2) - 1]


def filter_marker_y_padding(mrkr, padding, height):
    assert height > padding
    return mrkr[(mrkr > padding) & (mrkr < height - padding)]


def get_markers(a, avg_smoothed, padding, spacing=3):
    a0 = a[:-1]
    a0[0: padding] = avg_smoothed[0: padding]
    a1 = a[1:]
    a1[-padding:-1] = avg_smoothed[-padding:-1]

    a0 = smooth(a0, window_len=3)
    a1 = smooth(a1, window_len=3)
    id_up = np.where((a0 < (avg_smoothed - spacing)) & (a1 > avg_smoothed + spacing))[0]
    logger.debug("id_up befor filtering = %s ", len(id_up))

    id_up = filter_marker_y_padding(id_up, conf.marker_y_padding, len(a) - 1)
    logger.debug("id_up after filtering = %s ", len(id_up))

    logger.debug("id_up = %s", len(id_up))

    id_down = np.where((a0 > (avg_smoothed + spacing)) & (a1 < (avg_smoothed - spacing)))[0]
    logger.debug("id_down befor filtering = %s ", len(id_down))

    id_down = filter_marker_y_padding(id_down, conf.marker_y_padding, len(a) - 1)
    logger.debug("id_down after filtering = %s ", len(id_down))

    logger.debug("id_do = %s ", len(id_down))
    if len(id_up) != len(id_down):
        logger.debug("in_up = %s", id_up)
        logger.debug("in_down = %s", id_down)
        return np.array([])

    return np.stack((id_up, id_down), axis=1)


if __name__ == '__main__':

    file_path = '../data/colored/5.jpg'
    img = cv2.imread(file_path, 0)
    # plt.imshow(img, 'gray')
    # plt.show()
    height, width = img.shape
    logger.debug("height %s, width %s", height, width)


    vertices = get_four_corners(img)
    v_shifted = add_shift(vertices, conf.l_shift)
    sheet = transform(img, v_shifted, conf.rshape, False)
    # plt.subplot(121), plt.imshow(sheet, 'gray'), plt.title('Input')
    # plt.subplot(122), plt.imshow(sheet, 'gray'), plt.title('Sheet')
    # plt.show()
    # cv2.imwrite('../data/out/sheet.jpg', sheet)
    sheet_name = conf.sec_name.crop(sheet)
    sheet_type = conf.sec_type.crop(sheet)
    sheet_one = conf.sec_one.crop(sheet)
    sheet_marker = conf.sec_marker.crop(sheet)
    sheet_marker = marker_filter(sheet_marker)

    y_sum = sheet_marker.sum(1)
    # y_avg = np.average(y_sum)

    avg_smoothed = smooth(y_sum, window_len=conf.marker_smooth_window, window='flat')
    # print("avg = " + str(y_avg))
    logger.debug("y_sum shape = %s ", y_sum.shape)
    logger.debug("avg_s shape = %s ", avg_smoothed.shape)
    markers = get_markers(y_sum, avg_smoothed, conf.marker_y_padding, conf.marker_spacing)

    # for i in avg_smoothed:
    #     print(i)

    # for i in y_sum:
    #     print(i)

    # for yy, x_img in slide_marker(sheet_marker, conf.y_step, conf.y_window):
    #     plt.imshow(x_img, 'gray'), plt.title(str(yy))
    #     plt.show()

    for y0, y1 in markers:
        cv2.line(sheet, (0, y0), (width, y0), (0, 255, 255), 1)
        cv2.line(sheet, (0, y1), (width, y1), (0, 0, 255), 1)

    plt.subplot(231), plt.imshow(img, 'gray'), plt.title('Input')
    plt.subplot(232), plt.imshow(sheet, 'gray'), plt.title('Sheet')
    plt.subplot(233), plt.imshow(sheet_marker, 'gray'), plt.title('sheet_marker')
    plt.subplot(234), plt.imshow(sheet_name, 'gray'), plt.title('Name')
    plt.subplot(235), plt.imshow(sheet_type, 'gray'), plt.title('type')
    plt.subplot(236), plt.imshow(sheet_one, 'gray'), plt.title('one')
    plt.show()
    # border(img)
    # img_filtered = pre_filters(img)
    # findCorners(img_filtered)
    # print('done')
