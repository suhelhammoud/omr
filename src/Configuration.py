import numpy as np
import math


class Section:
    """region"""

    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def crop(self, img):
        return img[self.y0: self.y1, self.x0: self.x1]

    def translate(self, dx, dy):
        return Section(self.x0 + dx, self.y0 + dy,
                       self.x1 + dx, self.y1 + dy)


class OmrConfiguration:
    rshape = [1000, 1500]
    sec_name = Section(240, 25, 470, 270)
    sec_type = Section(470, 25, 550, 200)
    sec_answers = Section(15, 260, 500, 1270)
    sec_one = Section(15, 260, 265, 1270)
    sec_two = Section(260, 260, 500, 1270)
    y_step = 20
    y_window = 100
    marker_x0_bound = 0
    marker_x1_bound = 50
    # sec_marker = Section(0, 0, marker_r_shift - marker_l_shift, rshape[1])
    sec_marker = Section(marker_x0_bound, 0, marker_x1_bound, rshape[1])

    num_markers = 63
    marker_filter_blur = 3
    marker_y_padding_top = 40
    marker_y_padding_down = rshape[1] - 20
    marker_smooth_window = 150
    marker_threshold_spacing = 6
    marker_mx_height = 12  # TODO use in get+

    marker_height_range = range(6, 10)
    marker_space_range = range(20, 25)
    marker_width_range = range(7, 27)


    top_marker = Section(0, -5, 300, 15)


conf = OmrConfiguration


class Marker:
    def __init__(self, y0, y1):
        assert y1 > y0
        self.y0 = y0
        self.y1 = y1

    def height(self):
        return self.y1 - self.y0

    def is_in_h_range(self, h_r=conf.marker_height_range):
        return (self.y1 - self.y0) in h_r

    def is_lower_than(self, that):
        return self.x0 > that.x1

    def is_in_h_space(self, that, space=conf.marker_space_range):
        upper, lower = Marker.upper_lower(self, that)

        return (lower.y0 - upper.y0) in space \
               and (lower.y1 - upper.y1) in space

    @staticmethod
    def upper_lower(m1, m2):
        if m2.is_lower_than(m1):
            return m1, m2
        else:
            return m2, m1

    @staticmethod
    def can_acept(y0, y1):
        return y0 > conf.marker_y_padding_top \
               and y1 < conf.marker_y_padding_down \
               and y1 - y0 in conf.marker_height_range
