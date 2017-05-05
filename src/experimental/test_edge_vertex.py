from vertex_edge import Vertex, EDGE

import unittest


class EdgeVertexTestCase(unittest.TestCase):
    def test_vertex(self):
        c = (100, 100)  # center
        tl = (0, 0)  # top_left
        bl = (50, 150)  # bottom_left
        br = (150, 150)  # bottom_right
        tr = (150, 50)  # top_right

        # vertices
        self.assertEqual(Vertex.get(tl, c), Vertex.TOP_LEFT)
        self.assertEqual(Vertex.get(bl, c), Vertex.BOTTOM_LEFT)
        self.assertEqual(Vertex.get(br, c), Vertex.BOTTOM_RIGHT)
        self.assertEqual(Vertex.get(tr, c), Vertex.TOP_RIGHT)

        # r1 = EDGE.get([tl, tl, tr], c)
        # print(r1)
        # # print( [EDGE.str(i) for i in r1])

        # edges
        self.assertEqual(EDGE.get([tl, tl, tr], c), [EDGE.TOP])
        self.assertEqual(EDGE.get([tl, tr, tr], c), [EDGE.TOP])

        self.assertEqual(EDGE.get([tl, tl, br], c), [EDGE.TOP, EDGE.RIGHT])
        self.assertEqual(EDGE.get([tl, tr, br], c), [EDGE.TOP, EDGE.RIGHT])

        self.assertEqual(EDGE.get([tl, bl, br], c), [EDGE.LEFT, EDGE.BOTTOM])
        self.assertEqual(EDGE.get([tl, br, br], c), [EDGE.LEFT, EDGE.BOTTOM])

        self.assertEqual(EDGE.get([tl, bl, tr], c), [EDGE.LEFT, EDGE.TOP, EDGE.RIGHT])
        self.assertEqual(EDGE.get([tl, br, tr], c), [EDGE.LEFT, EDGE.TOP, EDGE.RIGHT])

        self.assertEqual(EDGE.get([bl, bl, br], c), [EDGE.BOTTOM])
        self.assertEqual(EDGE.get([bl, br, br], c), [EDGE.BOTTOM])

        self.assertEqual(EDGE.get([bl, tl, tr], c), [EDGE.LEFT, EDGE.TOP])
        self.assertEqual(EDGE.get([bl, tr, tr], c), [EDGE.LEFT, EDGE.TOP])

        self.assertEqual(EDGE.get([bl, bl, tr], c), [EDGE.BOTTOM, EDGE.RIGHT])
        self.assertEqual(EDGE.get([bl, br, tr], c), [EDGE.BOTTOM, EDGE.RIGHT])

        self.assertEqual(EDGE.get([bl, tl, br], c), [EDGE.LEFT, EDGE.TOP, EDGE.RIGHT])
        self.assertEqual(EDGE.get([bl, tr, br], c), [EDGE.LEFT, EDGE.TOP, EDGE.RIGHT])
