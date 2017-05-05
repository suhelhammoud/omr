import cv2
import numpy as np
from matplotlib import pyplot as plt


def find_draw_contours(img, org):
    img_c = img.copy()
    (_, cnts, _) = cv2.findContours(img_c, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    screenCnt = []
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, .03 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt.append(approx)

    for i in range(len(screenCnt)):
        cnt = screenCnt[i]
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_c, (x, y), (x + w, y + h), (255, 255, 100), 2)

    cv2.drawContours(org, screenCnt, -1, (0, 255, 0), 4)
    # cv2.imshow("org", org)
    cv2.imshow("img c", img_c)
    # cv2.imwrite("data/out/saved_org.jpg", org)
    print(screenCnt)
    print(len(screenCnt))
    return img_c


def circles(img):
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 1)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    return cimg


def lines(img, org):
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    minLineLength = 20
    maxLineGap = 50
    lines = cv2.HoughLines(edges, 1, np.pi / 90, 200)

    for x1, y1, x2, y2 in lines[0]:
        cv2.line(org, (x1, y1), (x2, y2), (255, 0, 0), 1)

    return edges

def forrieh(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    plt.subplot(121), plt.imshow(img, cmap= 'gray')
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.show()


if __name__ == '__main__':
    img = cv2.imread('../data/out/sections/sec_name.jpg', cv2.COLOR_BGR2GRAY)

    forrieh(img)

    if True:
        exit(0)

    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    # thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
    #                                cv2.THRESH_BINARY, 11, 0)


    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, okernel)
    #
    # kernel = np.ones((5, 5), np.uint8)
    # erosion = cv2.dilate(opening, kernel, iterations=1)


    plt.subplot(221), plt.imshow(img, cmap='gray')
    plt.subplot(222), plt.imshow(laplacian, cmap='gray')
    plt.subplot(223), plt.imshow(sobelx, cmap='gray')
    plt.subplot(224), plt.imshow(sobely, cmap='gray')

    # plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.show()
    #
    # find_draw_contours(img, img)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
