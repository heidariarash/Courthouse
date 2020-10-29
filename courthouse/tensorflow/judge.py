from courthouse.utils.case import CategoricalCase
import numpy as np
import tensorflow as tf

class CategoricalJudge:
    """
    Use this class to judge your model to see if it is fair or not.
    """
    def __init__(self) -> None:
        self.__org_data = None
        self.__new_data = None
        self.__old_case = None
        self.__new_case = None
        self.__org_out = None
        self.__new_out = None

    def case(self, data: np.ndarray, change_from: CategoricalCase , change_towards: CategoricalCase) -> None:
        """
        Use this method to specify
        """
        self.__old_case = change_from
        self.__new_case = change_towards
        self.__org_data = data[data[:, change_from.get("column")] == 1]
        self.__new_data = self.__org_data.copy()
        self.__new_data[:, change_from.get("column")] = 0
        self.__new_data[:, change_towards.get("column")] = 1

    def judge(self, model:tf.keras.Model, output_type: str) -> None:
        """
        Use this method to judge your model fairness.
        """
        org_predict = model.predict(self.__org_data)
        new_predict = model.predict(self.__org_data)
        if output_type == "categorical":
            self.__org_out = []
            self.__new_out = []

            for output in org_predict:
                self.__org_out.append(np.argmax(output))

            for output in new_predict:
                self.__new_out.append(np.argmax(output))

        if output_type == "binary":
            self.__org_out = []
            self.__new_out = []

            for output in org_predict:
                self.__org_out.append(1 if output>=0.5 else 0)

            for output in new_predict:
                self.__new_out.append(1 if output>=0.5 else 0)
            
    def verdict(self) -> str:
        pass
