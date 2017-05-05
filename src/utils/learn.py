
import cv2
import numpy as np;

def detect2(im):
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 100
    params.maxThreshold = 255

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 80

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
        print(kp)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 255, 0),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv2.imshow("Keypoints", im_with_keypoints)
    # cv2.waitKey(0)


def detect(im):
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 50
    params.maxThreshold = 150
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 150
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(im)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()