
import numpy as np


def smooth(x, window_len=150, window='flat'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.

    # Ref: http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """

    assert x.ndim == 1

    assert x.size > window_len

    if window_len < 3:
        return x

    assert window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    # return y
    return y[int(window_len / 2 - 1):-int(window_len / 2) - 1]


def get_crossing_downs_ups(values, avg, spacing=0):
    """
    Get the indexes of points in array "values" where they cross the avg line downward and upwards

    :param values: np.array(long)
    :param avg: int or np.array, fixed or moving average
    :param spacing: int, optional, trying to reduce noise
    :return: tuple(np.array, np.array),
        :id_down: indexes of crossing downward points,
        :id_up: indexes of crossing upward points,
    """
    a0 = values[:-1]
    a1 = values[1:]
    avg_plus = avg + spacing
    avg_minus = avg - spacing

    id_down = np.where((a0 > avg_plus)
                       & (a1 < avg_minus))[0]

    id_up = np.where((a0 < avg_minus)
                     & (a1 > avg_plus))[0]

    return id_down, id_up


def get_crossing_ups(a, avg, spacing=0):
    a0 = a[:-1]
    a1 = a[1:]
    id_up = np.where((a0 < (avg - spacing))
                     & (a1 > avg + spacing))[0]
    return id_up
