import urllib.request
import urllib
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from lpf import *
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        data = response.read()

    image = np.asarray(bytearray(data), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image


def urlRead():
    url = "http://192.168.1.107:8080/photo.jpg"

    while True:


        img = url_to_image(url)
        cv2.imshow('img', img)
        p = lpf.scan(img)

        if p is None:
            cv2.imshow('p', p)
        time.sleep(2)


    # cv2.destroyAllWindows()


def t1():
    # img = cv2.imread('data/out/sections/sec_name.jpg', cv2.COLOR_BGR2GRAY)
    img = cv2.imread('data/out/sections/sec_answers.jpg', cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                   cv2.THRESH_BINARY, 35, 0)

    dst = cv2.medianBlur(thresh,21)

    # kernel = np.ones((3, 3), np.uint8)
    # dst = cv2.dilate(dst, kernel, iterations=1)

    # dst = cv2.GaussianBlur(dst, (3, 3), 0)

    h = dst.sum(1)
    # print(len(h))
    for i in h:
        print(i)

    # plt.subplot(221)
    plt.imshow(dst, cmap='gray')
    plt.show()


if __name__ == '__main__':
    t1()
    # urlRead()
    pass
