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
