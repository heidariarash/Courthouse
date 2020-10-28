import pandas as pd
import numpy as np

class PandasManipulator():
    """
    Use this class to manipulate a Pandas dataframe and get new data.
    """
    def __init__(self):
        self.__column_change = ""
        self.__column_value = ""
        self.__column_old = None

    def config(self, **kwargs):
        """
        Use this method to configure the PandasManipulator settings.
        """
        for key, value in kwargs.items():
            if key == "column_to_change":
                self.__column_change = value

            elif key == "new_value":
                self.__column_value = value

            elif key == "old_value":
                self.__column_old = value

    def apply(self, dataframe):
        """
        Use this method to apply the manipulation on a dataframe
        """
        if (self.__column_change not in (dataframe.columns)):
            raise Exception(f"dataframe does not have a column named {self.__column_change}.")

        new_df = dataframe.copy()
        if self.__column_old == None:
            new_df[self.__column_change] = self.__column_value

        else:
            new_df[self.__column_change] = new_df[self.__column_change].replace(self.__column_old, self.__column_value)

        return new_df