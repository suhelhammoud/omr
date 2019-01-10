# METHOD #2: scikit-image
from skimage import io
import cv2
import matplotlib.pyplot as plt
import time

url = "http://192.168.1.181:8080/photoaf.jpg"
# loop over the image URLs
while True:
    # download the image using scikit-image
    print("downloading %s" % (url))
    image = io.imread(url)
    if image is not None:
        plt.subplot(111), plt.imshow(image, 'gray'), plt.title('Input')
        # plt.pause(1)
        plt.draw()
        # cv2.imshow("Incorrect", image)

    # time.sleep(1)
    # cv2.imshow("Correct", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break


