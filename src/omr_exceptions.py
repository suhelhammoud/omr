class VertexError(Exception):
    """
    Could not find vertex
    """
    pass


class BorderError(Exception):
    """
    Could not find four vertices
    """
    pass


class IDError(Exception):
    """
    Couldn't detect id
    """
    pass


class SheetTypeError(Exception):
    pass


class SheetNoTypeFoundError(Exception):
    pass


class MarkersNumberError(Exception):
    pass


class MarkerXError(Exception):
    pass


class MarkerCalibrateError(Exception):
    pass


class AnswerXBorderError(Exception):
    pass


class AnswerCalibrateError(Exception):
    pass


class AnswerMiddleError(Exception):
    pass
