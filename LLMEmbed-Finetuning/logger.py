"""
    This module contains the logger class.
"""


class Logger:
    """
    This class contains all the methods used for logging.
    """

    @classmethod
    def log(cls, message: str, enable_logging: bool) -> str:
        """
        This method displays the sent message.
        """
        if enable_logging:
            print(message)
