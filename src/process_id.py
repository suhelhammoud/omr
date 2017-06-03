import cv2
import numpy as np
from matplotlib import pyplot as plt


def otsu_id_filter(img):
    blur = cv2.medianBlur(img, 5, 0)
    # blur = img
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # kernel = np.ones((30, 30), np.uint8)
    # dilate = cv2.dilate(th3, kernel, iterations=1)
    dilate = th3
    return dilate


def process_id(id_marker, img):
    height, width = img.shape

    th3 = otsu_id_filter(img)
    sum_x = np.sum(th3, axis=0) / height
    sum_y = np.sum(th3, axis=1) / width

    plt.subplot(221), plt.imshow(th3, 'gray'), plt.title('th3')
    plt.subplot(222), plt.plot(sum_x, 'r'), plt.title('sum_x')
    plt.subplot(223), plt.plot(sum_y, 'b'), plt.title('sum_y')
    plt.subplot(224), plt.imshow(img, 'gray'), plt.title('img')
    plt.show()


if __name__ == '__main__':
    print("process name")
