class VertexError(Exception):
    """
    Could not find vertex
    """
    pass


class GetSideException(Exception):
    """
    Could not get H/V side of image
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
