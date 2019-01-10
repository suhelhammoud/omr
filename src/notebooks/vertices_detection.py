import cv2
import numpy as np
import logging

from matplotlib import pyplot as plt
from numpy.linalg import norm

from OmrExceptions import *

# from omr_utils import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def otsu_filter(img, blur_kernel=1):
    blur = cv2.medianBlur(img, blur_kernel, 0)  # TODO adjust the kernel
    _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def crop_to_four(image, h=None, w=None):
    if h is None:
        h = int(image.shape[0] / 2)
    if w is None:
        w = int(image.shape[1] / 2)
    # h, w = half_height_width(image)
    return image[0:h, 0: w], \
           image[0: h, w:], \
           image[h:, 0:w], \
           image[h:, w:]


def check_side(side_nz_array):
    return side_nz_array[-1] - side_nz_array[0] == len(side_nz_array) - 1


def get_side(image, axis=1):
    side = np.argmax(image, axis=axis)
    if np.any(image):
        nz = np.nonzero(side)[0]
        if len(nz) > 0 and nz[-1] - nz[0] == len(nz) - 1:
            return side, side[nz], nz
    raise GetSideException(f"Get Side Exception on axis={axis}")


def get_vertex(image, axis):
    side, side_nz, nz = get_side(image, axis)
    point = get_max_distant_point(nz, side_nz)
    return point[axis], point[1 - axis]


def vertex(image):
    for axis in [1, 0]:
        try:
            return get_vertex(image, axis=1)
        except GetSideException:
            logger.debug(f"GetSideException axis={axis}")
    raise GetSideException(f"Both V & H sides exceptions")


def normalize_quarters(tpl):
    return tpl[0], \
           np.fliplr(tpl[1]), \
           np.flipud(tpl[2]), \
           tpl[3][::-1, ::-1]
    # np.fliplr(np.flipud(tpl[3]))


def get_max_distant_point(x, y, a=None, b=None):
    """
    Given "ab" line segment and collection of ordered points with coordinates x and y, choose the point
    which has the maximum distant from the line passes through "ab",
    if "a" and "b" were not given then set "a" point to be the first point of the collection
    and "b" point to be the last one

    :param x: np.array([int,...int]), x coordinates of points.
    :param y: np.array([int,...int]), y coordinates of points.
    :param a: np.array([int, int]), optional, edge of line segment,
        if None then set it to the first point (x0, y0)
    :param b: np.array([int, int]), optional, edge of line segment,
        if None then set it to the last point (x0, y0)
    :return:np.array([int, int]) max distant point from "ab" segment
    """
    if len(x) == 0 or len(y) == 0:
        raise Exception(f"len(x) = {len(x)}, len(y) = {len(y)}")
    if a is None:
        try:
            a = np.array([x[0], y[0]])
        except:
            print(f"x = {x}, y = {y}")

    if b is None:
        b = np.array([x[-1], y[-1]])
    points = np.column_stack((x, y))
    distances = [distance(a, b, p) for p in points]
    mx_index = np.argmax(distances)
    return x[mx_index], y[mx_index]


def distance(a, b, p):
    """Distance between point "p" and line of "ab"

    :param a: np.array([int, int]) first edge a of segment line "ab".
    :param b: np.array([int, int]) second edge a of segment line "ab"
    :param p: np.array([int, int])
    :return: real, the distance from point "p" to line passes through "ab" segment
    """
    if all(a == p) or all(b == p):  # TODO unnecessary check, write tests to test optimization benefit of it
        return 0
    return norm(np.cross(b - a, a - p)) / norm(b - a)


def vertices_stacked(vrtcs, height, width, h=None, w=None):
    if h is None:
        h = int(height / 2)
    if w is None:
        w = int(width / 2)
    return [vrtcs[0],
            (width - vrtcs[1][0], vrtcs[1][1]),
            (vrtcs[2][0], height - vrtcs[2][1]),
            (width - vrtcs[3][0], height - vrtcs[3][1])]


def vertices(image):
    height, width = image.shape
    # h, w = int(height / 2), int(width / 2)
    # im_list = normalize_quarters(crop_to_four(image, h, w))
    im_list = normalize_quarters(crop_to_four(image))
    vrtcs = [vertex(im) for im in im_list]
    # return vertices_stacked(vrtcs, height, width, h, w)
    return vertices_stacked(vrtcs, height, width)


def transform(img, vertices, shape, show=False):
    """
     Wrap the region of answer sheet inside "img" into new image image with fized size defined in "shape"

    :param img: opencv image, gray, contains the answer sheet
    :param vertices: List, four point vertices of the answer sheet
    :param shape: array[width, height], size of resulting image
    :param show: boolean, for debugging purposes TODO delete later
    :return: opencv image, fixed size as in shape
    """
    pts1 = np.float32(vertices)

    # pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([[0, 0], [shape[0], 0], [0, shape[1]], shape])

    m_transform = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, m_transform, tuple(shape))

    if show:  # TODO delete later
        plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Input')
        plt.subplot(122), plt.imshow(dst, 'gray'), plt.title('Output')
        plt.show()
    return dst


def test():
    file_path = '../../data/in2/05.jpg'
    img = cv2.imread(file_path, 0)
    img_otsu = otsu_filter(img, blur_kernel=17)
    vtcs = vertices(img_otsu)
    print(vtcs)
    transform(img, vtcs, (1000, 1500), show=True)

    # plt.subplot(121), plt.imshow(img, 'gray'), plt.title('img')
    # plt.subplot(122), plt.imshow(img_otsu, 'gray'), plt.title('img_otsu')
    # plt.show()

    # im1, im2, im3, im4 = crop_to_four(img_otsu)
    im1, im2, im3, im4 = normalize_quarters(crop_to_four(img_otsu))

    # plt.subplot(221), plt.imshow(im1, 'gray'), plt.title('im1')
    # plt.subplot(222), plt.imshow(im2, 'gray'), plt.title('im2')
    # plt.subplot(223), plt.imshow(im3, 'gray'), plt.title('im3')
    # plt.subplot(224), plt.imshow(im4, 'gray'), plt.title('im4')
    # plt.show()


def old_expreiments():
    file_path = '../../data/in2/05.jpg'
    img = cv2.imread(file_path, 0)
    img_otsu = otsu_filter(img, blur_kernel=17)
    im = im1.copy()
    print(im.shape)
    ls, lsnz, lnz = get_side(im, axis=1)
    print(f"v test = {check_side(lnz)}, len = {len(lnz)}")

    print(f"ls zero = {ls[0]}, lsnz = {lsnz[0]}, lnz = {lnz[0]}")
    print(f"ls = {len(ls)}, lsnz ={len(lsnz)}, lnz={len(lnz)}")
    l_distance = get_max_distant_point(lnz, lsnz)
    print(f"l get_vertex = {get_vertex(im, axis=1)}")
    print(f"l_distance = {l_distance}")
    us, usnz, unz = get_side(im, axis=0)
    print(f"h test = {check_side(unz)}, len = {len(unz)}")

    u_distance = get_max_distant_point(unz, usnz)
    print(f"u_distance = {u_distance}")
    print(f"u get_vertex = {get_vertex(im, axis=0)}")

    cv2.circle(im, (l_distance[1], l_distance[0]), 50, (255, 255, 255), 3)
    cv2.circle(im, (u_distance[0], u_distance[1]), 70, (255, 255, 255), 4)

    plt.subplot(241), plt.imshow(im, 'gray'), plt.title('im1')
    plt.subplot(242), plt.plot(ls, 'b'), plt.title('v side')
    plt.subplot(243), plt.plot(lsnz, 'b'), plt.title('nz v side')
    plt.subplot(244), plt.plot(lnz, lsnz, 'b'), plt.title('lnz v side')

    plt.subplot(245), plt.imshow(im, 'gray'), plt.title('im1')
    plt.subplot(246), plt.plot(us, 'r'), plt.title('h side')
    plt.subplot(247), plt.plot(usnz, 'r'), plt.title('nz h side')
    plt.subplot(248), plt.plot(unz, usnz, 'r'), plt.title('xy')

    cv2.circle(im1, (u_distance[1], u_distance[0]), 50, (255, 255, 255), 3)
    plt.show()
    # print(f"h1 = {im1.shape[0]},"
    #       f"h3 = {im3.shape[0]}, "
    #       f"h1 + h3 = {im1.shape[0] + im3.shape[0]} "
    #       f"img.h = {img.shape[0]} ")


if __name__ == '__main__':
    test()
