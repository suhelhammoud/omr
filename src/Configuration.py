import numpy as np


class Section:
    """region"""

    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def crop(self, img):
        return img[self.y0: self.y1, self.x0: self.x1]


class OmrConfiguration:
    rshape = [1000, 1500]
    sec_name = Section(240, 25, 470, 270)
    sec_type = Section(470, 25, 550, 200)
    sec_answers = Section(15, 260, 500, 1270)
    sec_one = Section(15, 260, 265, 1270)
    sec_two = Section(260, 260, 500, 1270)
    l_shift = 100
    r_shift = 20
    sec_marker = Section(0, 0, l_shift + r_shift, rshape[1])
    y_step = 20
    y_window = 100

    marker_filter_blur = 7
    marker_y_padding = 50
    marker_smooth_window = 150
    marker_spacing = 2
