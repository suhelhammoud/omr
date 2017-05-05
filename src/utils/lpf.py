import cv2
import numpy as np
from pyimagesearch import imutils
from pyimagesearch.transform import four_point_transform
# from skimage.filters import threshold_adaptive
from matplotlib import pyplot as plt


class Section:
    """region"""

    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def crop(self, img):
        return img[self.y0: self.y1, self.x0: self.x1]


def saveRegions(img, sec, name):
    img_sec = sec.crop(img)
    cv2.imwrite("data/out/sections/" + name, img_sec)


def sobel(img):
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    return edged
    # cv2.imshow("edged", edged)
    # cv2.waitKey(0)


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

    # for i in range(len(screenCnt)):
    #     cnt = screenCnt[i]
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     cv2.rectangle(img_c, (x, y), (x + w, y + h), (255, 255, 100), 2)


    cv2.drawContours(org, screenCnt, -1, (0, 255, 0), 4)
    cv2.imshow("org", org)
    cv2.imwrite("data/out/saved_org.jpg", org)
    print(screenCnt)
    print(len(screenCnt))
    return img_c


def scan(image):
    screenCnt = None

    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    # gray = threshold_adaptive(gray, 251, offset=5)
    # warped = warped.astype("uint8") * 255

    edged = cv2.Canny(gray, 75, 200)

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # opencv3
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    # apply the four point transform to obtain a top-down
    # view of the original image
    if screenCnt is None:
        return None
    else:
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
        wraped_2 = cv2.resize(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), (1000, 1366))
        return wraped_2
        # cv2.imwrite("data/out/next.jpg", wraped_2)


def show(img):
    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def process(image_file):
    sec_name = Section(240, 25, 470, 270)
    sec_type = Section(470, 25, 550, 200)
    sec_answers = Section(15, 260, 500, 1270)
    sec_one = Section(15, 260, 265, 1270)
    sec_two = Section(260, 260, 500, 1270)

    img = cv2.imread(image_file)
    show(img)
    image = scan(img)
    if image is None:
        print("could not process image " + image_file)
        return
    show(image)

    cv2.imwrite("data/out/sections/sec_all.jpg", image)
    saveRegions(image, sec_name, "sec_name.jpg")
    saveRegions(image, sec_type, "sec_type.jpg")
    saveRegions(image, sec_answers, "sec_answers.jpg")
    saveRegions(image, sec_one, "sec_one.jpg")
    saveRegions(image, sec_two, "sec_two.jpg")


def edg(img):
    import cv2
    import numpy as np

    # filename = 'data/in/01.jpg'
    # img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imshow('dst', img)
    cv2.imwrite('data/out/edg.jpg', img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':

    process('/home/suhel/PycharmProjects/omr/data/in/03.jpg')

    # img = cv2.imread("data/out/next.jpg")
    # edg( img)
