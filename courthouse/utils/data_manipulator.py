import pandas as pd
import numpy as np

class PandasManipulator():
    """
    Use this class to manipulate a Pandas dataframe and get new data.
    """
    def __init__(self):
        self.__column_change = ""
        self.__new_value = ""
        self.__old_value = None

    def config(self, **kwargs):
        """
        Use this method to configure the PandasManipulator settings.
        """
        for key, value in kwargs.items():
            if key == "column_to_change":
                self.__column_change = value

            elif key == "new_value":
                self.__new_value = value

            elif key == "old_value":
                self.__old_value = value

    def apply(self, dataframe):
        """
        Use this method to apply the manipulation on a dataframe
        """
        if (self.__column_change not in (dataframe.columns)):
            raise Exception(f"dataframe does not have a column named {self.__column_change}.")

        new_df = dataframe.copy()
        if self.__old_value == None:
            new_df[self.__column_change] = self.__new_value

        else:
            new_df[self.__column_change] = new_df[self.__column_change].replace(self.__old_value, self.__new_value)

        return new_df


class NumpyNumericalManipulator():
    """
    Use this class to manipulate a Numpy array and get new data.
    """
    def __init__(self):
        self.__column_change = 0
        self.__new_value = ""
        self.__old_value = None

    def config(self, **kwargs):
        """
        Use this method to configure the NumpyNumericalManipulator settings.
        """
        for key, value in kwargs.items():
            if key == "column_to_change":
                self.__column_change = value

            elif key == "new_value":
                self.__new_value = value

            elif key == "old_value":
                self.__old_value = value

    def apply(self, array):
        """
        Use this method to apply the manipulation on a dataframe
        """
        new_arr = array.copy()
        if self.__old_value == None:
            new_arr[:,self.__column_change] = self.__new_value

        else:
            for i, item in enumerate(new_arr[:, self.__column_change]):
                if item == self.__old_value:
                    new_arr[i, self.__column_change] = self.__new_value

        return new_arr


class NumpyCategoricalManipulator():
    """
    Use this class to manipulate a Numpy Array and get new data.
    """
    def __init__(self):
        self.__old_columns = 0
        self.__new_column = 0

    def config(self, **kwargs):
        """
        Use this method to configure the NumpyCategoricalManipulator settings.
        """
        for key, value in kwargs.items():
            if key == "old_columns":
                self.__old_columns = value

            elif key == "new_column":
                self.__new_column = value

    def apply(self, array):
        """
        Use this method to apply the manipulation on a dataframe
        """
        new_arr = array.copy()
        
        for i, item in enumerate(new_arr[:, self.__old_columns]):
            if (item == 1).any():
                print(new_arr[i, self.__new_column])
                new_arr[i, self.__new_column] = 1
                new_arr[i, self.__old_columns] = 0

        return new_arr