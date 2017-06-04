import logging

from matplotlib import pyplot as plt

from OmrExceptions import *
from omr_utils import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

'''y_type_positions
    Calculated:
    1.000 (calibre), 1.577, 2.000, 2.423, 2.846
    Practical:
    1.00, 1.58, 2.05, 2.48, 2.9
'''


def process_type(img, debug=True):
    height, width = img.shape
    img_blob = otsu_filter(img, 7)  # TODO adjust kernel later

    sum_x = np.sum(img_blob, axis=0) / height
    x_downs, x_ups = get_crossing_downs_ups(sum_x, 240)
    if len(x_downs) != 1 or len(x_ups) != 1:
        logger.error("downs: %s, ups: %s", x_downs, x_ups)
        plt.subplot(121), plt.imshow(img, 'gray'), plt.title('img_x')  # TODO to be deleted later
        plt.subplot(122), plt.imshow(img_blob, 'gray'), plt.title('img_blob')
        plt.show()
        raise SheetTypeError("Could not locate the vertical borders of type box")

    x_center = int((x_downs[0] + x_ups[0]) / 2)

    sum_y = np.sum(img_blob, axis=1) / width
    y_downs, y_ups = get_crossing_downs_ups(sum_y, 240)
    if len(y_downs) != 2 or len(y_ups) != 2:
        logger.error("y_downs: %s, y_ups: %s", y_downs, y_ups)
        plt.subplot(121), plt.imshow(img, 'gray'), plt.title('img_x')  # TODO to be deleted later
        plt.subplot(122), plt.imshow(img_blob, 'gray'), plt.title('img_blob')
        plt.show()
        raise SheetNoTypeFoundError("No Type Found")

    y_blob = int((y_ups[1] + y_downs[1]) / 2)

    y_unit = y_ups[0] - y_downs[0]
    y_type_positions = [1.58, 2.05, 2.48, 2.9]
    y_centers = [int(i * y_unit) + y_downs[0] for i in y_type_positions]

    # Find the index of closest element in pre-defined y_centers to the newly located y_blob
    type_index, type_position = min(enumerate(y_centers), key=lambda x: abs(x[1] - y_blob))
    # Map index to corresponding sheet type
    sheet_type = {0: "A", 1: "B", 2: "C", 3: "D"}[type_index]
    logger.info('Sheet Type = %s' % sheet_type)

    if debug:
        vis = img.copy()
        cv2.line(vis, (x_center, 0), (x_center, height), (0, 0, 0), 1)
        for i in y_centers:
            cv2.line(vis, (0, i), (width, i), (0, 0, 0), 1)

        plt.subplot(221), plt.imshow(vis, 'gray'), plt.title('img')
        plt.subplot(222), plt.imshow(img_blob, 'gray'), plt.title('blob')
        plt.subplot(223), plt.plot(sum_y, 'g'), plt.title('sum_y')
        plt.axvline(x=y_blob)
        plt.subplot(224), plt.plot(sum_x, 'r'), plt.title('sum_x')
        plt.axvline(x=x_center)
        plt.show()

    return sheet_type

