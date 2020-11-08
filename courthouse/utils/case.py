from typing import Union

class CategoricalCase:
    """
    Use this class to specify a categorical case.
    """
    def __init__(self, name: str, column: Union[int, list, None] = None, binary = -1) -> None:
        self.__column = column
        self.__name = name
        self.__binary = binary

    def get(self, key: str) -> Union[str, int]:
        """
        This is a getter method for CategoricalCase class.
        """
        if key == "column":
            return self.__column

        elif key == "name":
            return self.__name

        elif key == "binary":
            return self.__binary


class NumericalCase:
    """
    Use this class to specify a numerical case.
    """
    def __init__(self, column: int, name: str) -> None:
        self.__column = column
        self.__name = name

    def get(self, key: str) -> Union[str, int]:
        """
        This is a getter method for NumericalCase.
        """
        if key == "column":
            return self.__column

        elif key == "name":
            return self.__name