
import numpy as np
from matplotlib import pyplot as plt


def foo( x_nz , scan_x_nz):
   y0 = scan_x_nz[0]
   y1 = scan_x_nz[-1]

   a = scan_x_nz - y0
   b = scan_x_nz - y1
   x = np.range(len(a))

   v0 = np.row_stack( (x, a))
   v1 = np.row_stack( (x, b))


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(rad)



if __name__ == '__main__':
    scan_x_nz = np.load('../data/pickles/a.npy')
    print(scan_x_nz)
    skip = 30
    x = np.arange(len(scan_x_nz))
    v = np.column_stack((x, scan_x_nz))
    p1 = v[0]
    p2 = v[-1]

    v1 = v - p1
    v2 = p2 - v
    print(v1.shape)
    print(v2.shape)

    ang = [angle_between(v1[i], v2[i]) for i in x[skip:-skip]]
    xmax = np.argmax(ang) + skip
    print(xmax)

    # plt.plot(v1[:,0], v1[:,1], 'r', v2[:,0], v2[:,1], 'g')
    plt.plot(ang)

    plt.show()
    print('done')