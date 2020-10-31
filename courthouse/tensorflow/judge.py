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
        self.__output_type = None

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
        self.__output_type = output_type
        if output_type == "categorical":
            self.__org_out = []
            self.__new_out = []

            for output in org_predict:
                self.__org_out.append(np.argmax(output))

            for output in new_predict:
                self.__new_out.append(np.argmax(output))

        if output_type == "binary_sigmoid":
            self.__org_out = []
            self.__new_out = []

            for output in org_predict:
                self.__org_out.append(1 if output>=0.5 else 0)

            for output in new_predict:
                self.__new_out.append(1 if output>=0.5 else 0)

        if output_type == "binary_tanh":
            self.__org_out = []
            self.__new_out = []

            for output in org_predict:
                self.__org_out.append(1 if output>=0 else 0)

            for output in new_predict:
                self.__new_out.append(1 if output>=0 else 0)

        else:
            self.__output_type = None
            raise Exception(f'{output_type} output_type is not defined.')
            
    def verdict(self) -> str:
        """
        Use this method to print the report of the fairness of the model.
        """
        #checking if the model is actually judged
        if self.__output_type is None:
            print('No model has been judged yet.')

        print(f"There are {self.__org_data.shape[0]} \"{self.__old_case.get('name')}\" to be found.\n")
        print("When the model was applied to the original dataset, these results where obtained:")
        if self.__output_type == "binary_sigmoid":
            ones = sum(filter(lambda x: x==1, self.__org_out))
            print(f"\t{ones} time(s) the model predict 1. This is the case for {ones/len(self.__org_out)*100}% of the data.")
            print(f"\t{len(self.__org_out) - ones} time(s) the model predict 0. This is the case for {(1 - ones/len(self.__org_out))* 100}% of the data.\n")

            print(f"Then dataset was changed in a way that all the {self.__old_case.get('name')} were changed to {self.__new_case.get('name')}.\n")
            print("These results were obtained after applying the model on the new data.")

            ones = sum(filter(lambda x: x==1, self.__new_out))
            print(f"\t{ones} time(s) the model predict 1. This is the case for {ones/len(self.__new_out) * 100}% of the data.")
            print(f"\t{len(self.__new_out) - ones} time(s) the model predict 0. This is the case for {(1 - ones/len(self.__new_out)) * 100}% of the data.")

