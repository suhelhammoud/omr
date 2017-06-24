import numpy as np


def answer_blob_coverage(img, x, y,
                         delta=(4, 3),
                         threshold=None):
    """
    Percentage of marking coverage. Ideally, if marked, then coverage = 1.0 (black),
    if not marked at all then coverage = 0.0 (white)

    :param img: roi contains roughly the id_box
    :param x: int, blob x coordinates
    :param y: int, blob y coordinates
    :param delta: tuple(int,int), blob scan diameter (dx,dy)
    :param threshold: float (optional), used to return either 0.0 or 1.0 values
    :return: marking percentage, value between 1.0 (fully marked) and 0.0 (not marked).
        If threshold is applied then return either 1.0 (fully marked) or 0.0 (not marked)
    """
    delta_y = delta[0]
    delta_x = delta(1)
    norm = 4 * delta_x * delta_y * 255
    covarage = 1 - (np.sum(img[y - delta_y: y + delta_y, x - delta_x: x + delta_x]) / norm)  # inverse percent level
    if threshold is None:
        return covarage
    elif covarage >= threshold:
        return 1
    else:
        return 0
        # return 1 if covarage >= threshold else 0


class Answer:
    X = [.193, .372, .55, .729, .908]

    def __init__(self, num, x1, x2, y1, y2, height=20):
        self.num = num
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        self.width = self.x2 - self.x1
        self.height = height
        self.y_shift = (self.y2 - self.y1) / self.width
        self.choice_x = [int(round(xr * self.width)) + x1 for xr in Answer.X]
        self.choice_y = [int(round(x * self.y_shift)) + y1 for x in self.choice_x]

    def choices(self):
        return zip(self.choice_x, self.choice_y)

    def __str__(self):
        return "Answer { num:%s, color=%s }" % (self.num, self.color())

    def color(self):
        ['GRAY', 'WHITE'][id % 2]

    def crop(self, sheet):
        return sheet[self.y - self.height: self.y + self.height, self.x1: self.x2]

    def mark(self, img, threshold=None):
        return [answer_blob_coverage(img, xi, yi, (4, 3), threshold=threshold)
                for xi, yi in self.choices()]
