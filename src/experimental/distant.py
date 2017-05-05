import numpy as np
import cv2
from matplotlib import pyplot as plt
# from skimage.filters import threshold_adaptive
from learn import detect, detect2

def applyMor(gray):
    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    return gray


def centers():
    # img = cv2.imread('data/out/next.jpg')
    img = cv2.imread('data/out/sections/sec_answers.jpg')
    # img = cv2.imread('data/out/sections/sec_two.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray =cv2.medianBlur(gray, 9 , 0)
    # kernel = np.ones((1, 1), np.uint8)
    # gray = cv2.dilate(gray, kernel, iterations=1)

    cv2.imshow('grabi', gray)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                   cv2.THRESH_BINARY, 29, 0)
    #
    # # noise removal
    # kernel = np.ones((2, 2), np.uint8)
    #
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations= 1)
    #
    # # sure background area
    # sure_bg = opening
    # sure_bg = cv2.dilate(opening, kernel, iterations=1)
    # #
    # # Finding sure foreground area
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)

    cv2.imshow('org', img)
    cv2.imshow("sure_fg", sure_fg)
    inverted = cv2.bitwise_not(sure_fg)
    # inverted =cv2.medianBlur(inverted, 5 , 0)

    kernel = np.ones((3, 3), np.uint8)
    inverted = cv2.dilate(inverted, kernel, iterations=1)

    cv2.imshow("inverted", inverted)
    # blob(inverted)

    cv2.imwrite("data/out/inverted.jpg", inverted)

    blob(inverted)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def blob(im):
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 100
    params.maxThreshold = 255

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100

    # Filter by Circularity
    # params.filterBCircularity = True
    # params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.1

    # Filter by Inertia
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(im)

    for kp in keypoints:
        print(str(kp.pt[0] )+ "\t" + str(kp.pt[1]) +"\t" +str(kp.size))
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 255, 0),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.imwrite("data/out/keypoints.jpg", im_with_keypoints)

    # cv2.waitKey(0)





def test():
    # img = cv2.imread('data/out/sections/sec_one.jpg')
    img = cv2.imread('data/out/next.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # warped = threshold_adaptive(gray, 401, offset = 5)
    # warped = warped.astype("uint8") * 255

    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 29, 0)
    th2 = applyMor(th2)
    #
    # th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #             cv2.THRESH_BINARY,29,0)

    cv2.imshow('gray', gray)
    cv2.imshow('th2', th2)
    # cv2.imshow('th3', th3)

    cv2.waitKey(0)

    #


# # noise removal
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# # sure background area
# sure_bg = cv2.dilate(opening,kernel,iterations=3)
# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)
#

if __name__ == '__main__':
    centers()
