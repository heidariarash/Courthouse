from typing import Union

class CategoricalCase():
    """
    Use this class to specify a categorical case.
    """
    def __init__(self, column: int, name: str) -> None:
        self.__column = column
        self.__name = name

    def get(self, key: str) -> Union[str, int]:
        """
        Thie is a getter method for CategoricalCase class.
        """
        if key == "column":
            return self.__column

        elif key == "name":
            return self.__name