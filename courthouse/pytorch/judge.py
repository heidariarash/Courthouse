from courthouse.utils.case import CategoricalCase
import numpy as np
import torch
import torch.nn as nn

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
        print(self.__org_data)
        print(self.__new_data)

    def judge(self, model:nn.Module, output_type: str) -> None:
        """
        Use this method to judge your model fairness.
        """
        org_predict = model(torch.from_numpy(self.__org_data).type('torch.FloatTensor'))
        new_predict = model(torch.from_numpy(self.__new_data).type('torch.FloatTensor'))
        self.__output_type = output_type
        if output_type == "categorical":
            self.__org_out = []
            self.__new_out = []

            for output in org_predict:
                self.__org_out.append(np.argmax(output))

            for output in new_predict:
                self.__new_out.append(np.argmax(output))

        elif output_type == "binary_sigmoid":
            self.__org_out = []
            self.__new_out = []

            for output in org_predict:
                self.__org_out.append(1 if output>=0.5 else 0)

            for output in new_predict:
                self.__new_out.append(1 if output>=0.5 else 0)

        elif output_type == "binary_tanh":
            self.__org_out = []
            self.__new_out = []

            for output in org_predict:
                self.__org_out.append(1 if output>=0 else 0)

            for output in new_predict:
                self.__new_out.append(1 if output>=0 else 0)

        elif output_type == "binary_with_logits":
            org_predict = torch.sigmoid(org_predict)
            new_predict = torch.sigmoid(new_predict)
            self.__org_out = []
            self.__new_out = []

            for output in org_predict.data:
                self.__org_out.append(1 if output>=0.5 else 0)

            for output in new_predict.data:
                self.__new_out.append(1 if output>=0.5 else 0)

        elif output_type == "regression":
            self.__org_out = []
            self.__new_out = []

            self.__org_out.append(torch.mean(org_predict).data.numpy())
            self.__org_out.append(torch.min(org_predict).data.numpy())
            self.__org_out.append(torch.max(org_predict).data.numpy())
            self.__new_out.append(torch.mean(new_predict).data.numpy())
            self.__new_out.append(torch.min(new_predict).data.numpy())
            self.__new_out.append(torch.max(new_predict).data.numpy())

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
        if self.__output_type == "binary_sigmoid" or self.__output_type == "binary_tanh" or self.__output_type == "binary_with_logits":
            ones = sum(filter(lambda x: x==1, self.__org_out))
            print(f"\t{ones} time(s) the model predicted 1. This is the case for {ones/len(self.__org_out)*100}% of the data.")
            print(f"\t{len(self.__org_out) - ones} time(s) the model predicted 0. This is the case for {(1 - ones/len(self.__org_out))* 100}% of the data.\n")

            print(f"Then dataset was changed in a way that all the {self.__old_case.get('name')} were changed to {self.__new_case.get('name')}.\n")
            print("These results were obtained after applying the model on the new data.")

            ones = sum(filter(lambda x: x==1, self.__new_out))
            print(f"\t{ones} time(s) the model predicted 1. This is the case for {ones/len(self.__new_out) * 100}% of the data.")
            print(f"\t{len(self.__new_out) - ones} time(s) the model predicted 0. This is the case for {(1 - ones/len(self.__new_out)) * 100}% of the data.")

        elif self.__output_type == "categorical":
            results = {}
            for output in self.__org_out:
                results[output] = results.get(output, 0) + 1
            
            for key, value in results.items():
                print(f"\t{value} time(s) the model predicted {key}. This is the case for {value/len(self.__org_out)*100}% of the data.")
            print("\n")

            print(f"Then dataset was changed in a way that all the {self.__old_case.get('name')} were changed to {self.__new_case.get('name')}.\n")
            print("These results were obtained after applying the model on the new data.")

            results = {}
            for output in self.__new_out:
                results[output] = results.get(output, 0) + 1
            
            for key, value in results.items():
                print(f"\t{value} time(s) the model predicted {key}. This is the case for {value/len(self.__org_out)*100}% of the data.")

        elif self.__output_type == "regression":
            print(f"\tMean of the predictions: {self.__org_out[0]}")
            print(f"\tMinimum of the predictions: {self.__org_out[1]}")
            print(f"\tMaximum of the predictions: {self.__org_out[2]}\n")
            print(f"Then dataset was changed in a way that all the {self.__old_case.get('name')} were changed to {self.__new_case.get('name')}.\n")
            print("These results were obtained after applying the model on the new data.")
            print(f"\tMean of the predictions: {self.__new_out[0]}")
            print(f"\tMaximum of the predictions: {self.__new_out[1]}")
            print(f"\tMaximum of the predictions: {self.__new_out[2]}")

    def faced_discrimination(self) -> list:
        """
        Use this method to get a list of datapoints, for which the prediction would be different if the case was different.
        """
        if self.__output_type == "regression":
            print("You can not use this method on a regression problem.")
            return
            
        differnet = []
        for i, output in enumerate(self.__org_out):
            if output != self.__new_out[i]:
                differnet.append(self.__org_data[i])

        return differnet


class NumericalJudge:
    """
    Use this class to judge your model to see if it is fair or not.
    """
    def __init__(self) -> None:
        self.__org_data = None
        self.__new_data = None
        self.__case = None
        self.__org_out = None
        self.__new_out = None
        self.__output_type = None

    def case(self, case:NumericalCase, data: np.ndarray, change_amount: int) -> None:
        """
        Use this method to specify
        """
        self.__case = case
        self.__org_data = data
        self.__new_data = data.copy()
        self.__new_data[:, case.get("column")] = self.__new_data[:, case.get("column")] + change_amount

    def judge(self, model:nn.Module, output_type: str) -> None:
        """
        Use this method to judge your model fairness.
        """
        org_predict = model(torch.from_numpy(self.__org_data).type('torch.FloatTensor'))
        new_predict = model(torch.from_numpy(self.__new_data).type('torch.FloatTensor'))
        print(self.__new_data)
        print(self.__org_data)
        self.__output_type = output_type
        if output_type == "categorical":
            self.__org_out = []
            self.__new_out = []

            for output in org_predict:
                self.__org_out.append(np.argmax(output))

            for output in new_predict:
                self.__new_out.append(np.argmax(output))

        elif output_type == "binary_sigmoid":
            self.__org_out = []
            self.__new_out = []

            for output in org_predict:
                self.__org_out.append(1 if output>=0.5 else 0)

            for output in new_predict:
                self.__new_out.append(1 if output>=0.5 else 0)

        elif output_type == "binary_tanh":
            self.__org_out = []
            self.__new_out = []

            for output in org_predict:
                self.__org_out.append(1 if output>=0 else 0)

            for output in new_predict:
                self.__new_out.append(1 if output>=0 else 0)

        elif output_type == "binary_with_logits":
            org_predict = torch.sigmoid(org_predict)
            new_predict = torch.sigmoid(new_predict)
            self.__org_out = []
            self.__new_out = []

            for output in org_predict.data:
                self.__org_out.append(1 if output>=0.5 else 0)

            for output in new_predict.data:
                self.__new_out.append(1 if output>=0.5 else 0)

        elif output_type == "regression":
            self.__org_out = []
            self.__new_out = []

            self.__org_out.append(torch.mean(org_predict).data.numpy())
            self.__org_out.append(torch.min(org_predict).data.numpy())
            self.__org_out.append(torch.max(org_predict).data.numpy())
            self.__new_out.append(torch.mean(new_predict).data.numpy())
            self.__new_out.append(torch.min(new_predict).data.numpy())
            self.__new_out.append(torch.max(new_predict).data.numpy())

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

        print(self.__new_out)
        print(self.__org_out)

        print(f"There are {self.__org_data.shape[0]} datapoint in original dataset.\n")
        print("When the model was applied to the original dataset, these results where obtained:")
        if self.__output_type == "binary_sigmoid" or self.__output_type == "binary_tanh" or self.__output_type == "binary_with_logits":
            ones = sum(filter(lambda x: x==1, self.__org_out))
            print(f"\t{ones} time(s) the model predicted 1. This is the case for {ones/len(self.__org_out)*100}% of the data.")
            print(f"\t{len(self.__org_out) - ones} time(s) the model predicted 0. This is the case for {(1 - ones/len(self.__org_out))* 100}% of the data.\n")

            print(f"Then the value of {self.__case.get('name')} changed.")
            print("\n")
            print("These results were obtained after applying the model on the new data.")

            ones = sum(filter(lambda x: x==1, self.__new_out))
            print(f"\t{ones} time(s) the model predicted 1. This is the case for {ones/len(self.__new_out) * 100}% of the data.")
            print(f"\t{len(self.__new_out) - ones} time(s) the model predicted 0. This is the case for {(1 - ones/len(self.__new_out)) * 100}% of the data.")

        elif self.__output_type == "categorical":
            results = {}
            for output in self.__org_out:
                results[output] = results.get(output, 0) + 1
            
            for key, value in results.items():
                print(f"\t{value} time(s) the model predicted {key}. This is the case for {value/len(self.__org_out)*100}% of the data.")
            print("\n")

            print(f"Then the value of {self.__case.get('name')} changed.")
            print("\n")
            print("These results were obtained after applying the model on the new data.")

            results = {}
            for output in self.__new_out:
                results[output] = results.get(output, 0) + 1
            
            for key, value in results.items():
                print(f"\t{value} time(s) the model predicted {key}. This is the case for {value/len(self.__org_out)*100}% of the data.")
                
        elif self.__output_type == "regression":
            print(f"\tMean of the predictions: {self.__org_out[0]}")
            print(f"\tMinimum of the predictions: {self.__org_out[1]}")
            print(f"\tMaximum of the predictions: {self.__org_out[2]}\n")
            print(f"Then the value of {self.__case.get('name')} changed.")
            print("\n")
            print("These results were obtained after applying the model on the new data.")
            print(f"\tMean of the predictions: {self.__new_out[0]}")
            print(f"\tMaximum of the predictions: {self.__new_out[1]}")
            print(f"\tMaximum of the predictions: {self.__new_out[2]}")

    def faced_discrimination(self) -> list:
        """
        Use this method to get a list of datapoints, for which the prediction would be different if the case was different.
        """
        if self.__output_type == "regression":
            print("You can not use this method on a regression problem.")
            return
            
        differnet = []
        for i, output in enumerate(self.__org_out):
            if output != self.__new_out[i]:
                differnet.append(self.__org_data[i])

        return differnet