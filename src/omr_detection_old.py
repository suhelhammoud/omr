import numpy as np
import cv2
from matplotlib import pyplot as plt

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



def law_of_cosines(a, x, b):
    xa = a - x
    xc = b - x
    # calculate angle
    cosine_angle = np.dot(xa, xc) / (np.linalg.norm(xa) * np.linalg.norm(xc))

    angle = np.arccos(cosine_angle)
    return angle
    # pAngle = np.degrees(angle)


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



if __name__ == '__main__':
    file_path = '../data/colored/3.jpg'
    img = cv2.imread(file_path, 0)
    border(img)