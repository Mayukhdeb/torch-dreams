class PytorchVersionError(Exception):
    """Raised when the user is not on pytorch 1.8.x or higher

    Args:
        version (str): torch.__version__
    """

    def __init__(self, version):
        self.version = version
        self.message = (
            "Expected pytorch to have version 1.8.x or higher but got: "
            + self.version
            + "\n Please consider updating pytorch from: https://pytorch.org/get-started/locally"
        )

    def __str__(self):
        return self.message
