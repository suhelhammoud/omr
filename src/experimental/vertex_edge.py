class Page:
    def __init__(self, tl, tr, bl, br):
        self.top_left = tl
        self.top_right = tr
        self.bottom_left = bl
        self.bottom_right = br

    def set_edge(self, edge, edge_type):
        if edge_type == EDGE.LEFT:
            self.left = edge
        elif edge_type == EDGE.TOP:
            self.top = edge
        elif edge_type == EDGE.RIGHT:
            self.right = edge
        elif edge_type == EDGE.BOTTOM:
            self.bottom = edge
        else:
            print(" Error not found edge_type " + edge_type)

    def set_vertex(self, point, v_type):
        if v_type == Vertex.TOP_LEFT:
            self.top_left = point
        elif v_type == Vertex.TOP_RIGHT:
            self.top_right = point
        elif v_type == Vertex.BOTTOM_LEFT:
            self.bottom_left = point
        elif v_type == Vertex.BOTTOM_RIGHT:
            self.bottom_right = point
        else:
            print(" Error not found v_type " + v_type)


class Vertex:
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3

    __str_map = {
        TOP_LEFT: "TOP_LEFT",
        TOP_RIGHT: "TOP_RIGHT",
        BOTTOM_LEFT: "BOTTOM_LEFT",
        BOTTOM_RIGHT: "BOTTOM_RIGHT",
    }

    @staticmethod
    def str(v):
        return Vertex.__str_map.get(v, "NOT_FOUND")

    @staticmethod
    def get(point, center):
        assert point[0] != center[0]
        assert point[1] != center[1]

        if point[0] < center[0]:  # LEFT
            if point[1] < center[1]:
                return Vertex.TOP_LEFT
            else:
                return Vertex.BOTTOM_LEFT
        else:
            if point[1] < center[1]:
                return Vertex.TOP_RIGHT
            else:
                return Vertex.BOTTOM_RIGHT

    @staticmethod
    def is_top(v):
        return v == Vertex.TOP_RIGHT or v == Vertex.TOP_LEFT

    @staticmethod
    def is_bottom(v):
        return not Vertex.is_top(v)

    @staticmethod
    def is_left(v):
        return v == Vertex.TOP_LEFT or v == Vertex.BOTTOM_LEFT

    @staticmethod
    def is_right(v):
        return not Vertex.is_left(v)


class EDGE:
    TOP = 0
    LEFT = 1
    RIGHT = 2
    BOTTOM = 3

    __str_map = {
        TOP: "TOP",
        LEFT: "LEFT",
        RIGHT: "RIGHT",
        BOTTOM: "BOTTOM"
    }

    __mapper = {
        (Vertex.TOP_LEFT, Vertex.TOP_LEFT, Vertex.TOP_RIGHT): [TOP],  # ---
        (Vertex.TOP_LEFT, Vertex.TOP_RIGHT, Vertex.TOP_RIGHT): [TOP],  # ---

        (Vertex.TOP_LEFT, Vertex.TOP_LEFT, Vertex.BOTTOM_RIGHT): [TOP, RIGHT],  # --\
        (Vertex.TOP_LEFT, Vertex.TOP_RIGHT, Vertex.BOTTOM_RIGHT): [TOP, RIGHT],  # --\

        (Vertex.TOP_LEFT, Vertex.BOTTOM_LEFT, Vertex.BOTTOM_RIGHT): [LEFT, BOTTOM],  # \__
        (Vertex.TOP_LEFT, Vertex.BOTTOM_RIGHT, Vertex.BOTTOM_RIGHT): [LEFT, BOTTOM],  # \__

        (Vertex.TOP_LEFT, Vertex.BOTTOM_LEFT, Vertex.TOP_RIGHT): [LEFT, TOP, RIGHT],  # \__/
        (Vertex.TOP_LEFT, Vertex.BOTTOM_RIGHT, Vertex.TOP_RIGHT): [LEFT, TOP, RIGHT],  # \__/

        (Vertex.BOTTOM_LEFT, Vertex.BOTTOM_LEFT, Vertex.BOTTOM_RIGHT): [BOTTOM],  # ___
        (Vertex.BOTTOM_LEFT, Vertex.BOTTOM_RIGHT, Vertex.BOTTOM_RIGHT): [BOTTOM],  # ___

        (Vertex.BOTTOM_LEFT, Vertex.TOP_LEFT, Vertex.TOP_RIGHT): [LEFT, TOP],  # /--
        (Vertex.BOTTOM_LEFT, Vertex.TOP_RIGHT, Vertex.TOP_RIGHT): [LEFT, TOP],  # /--

        (Vertex.BOTTOM_LEFT, Vertex.BOTTOM_LEFT, Vertex.TOP_RIGHT): [BOTTOM, RIGHT],  # __/
        (Vertex.BOTTOM_LEFT, Vertex.BOTTOM_RIGHT, Vertex.TOP_RIGHT): [BOTTOM, RIGHT],  # __/

        (Vertex.BOTTOM_LEFT, Vertex.TOP_LEFT, Vertex.BOTTOM_RIGHT): [LEFT, TOP, RIGHT],  # /--\
        (Vertex.BOTTOM_LEFT, Vertex.TOP_RIGHT, Vertex.BOTTOM_RIGHT): [LEFT, TOP, RIGHT],  # /--\

    }

    _tmp = {}
    for key, value in __mapper.items():
        _tmp[key[::-1]] = value
    __mapper.update(_tmp)

    @staticmethod
    def get_three_points(points_x, points_y):
        half_len = int(len(points_x) / 2)
        y2 = points_y[half_len]  # get the y2 of middle side
        x2 = points_x[half_len]
        p2 = (x2, y2)
        p1 = (points_x[0], points_y[0])
        p3 = (points_x[-1], points_y[-1])
        return [p1, p2, p3]

    @staticmethod
    def get(three_points, center):
        assert len(three_points) == 3
        p1 = three_points[0]
        p2 = three_points[1]
        p3 = three_points[2]

        v1 = Vertex.get(p1, center)
        v2 = Vertex.get(p2, center)
        v3 = Vertex.get(p3, center)
        key = (v1, v2, v3)
        return EDGE.__mapper.get(key)

    @staticmethod
    def str(edge):
        return EDGE.__str_map.get(edge, "edge not found")

    @staticmethod
    def str_array(edges):
        return [EDGE.str(i) for i in edges]
