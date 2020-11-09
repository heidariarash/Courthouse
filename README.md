# Courthouse
Courthouse is an AI fairness analysis library for machine learning models. This model supports both TensorFlow and PyTorch.

## Installation
To install the library, use pip.
```bash
pip install courthouse
```

## Preparing Data for Courthouse
You need data to train the TensorFlow/PyTorch model. You also need to pass the data to the Courthouses methods. Courthouse expects the data in its numpy.ndarray format, i.e. if your data is in other formats such as pandas.DataFrame or Tensors, you should convert the data to numpy.
```python
import numpy as np

data = np.array(...)
```

There are two Judges in Courthouse. One for categorical cases (one-hot encoded inputs), and the other is for numerical cases (guess what? not one-hot encoded columns).

For example, assume that your dataset has a `sex` column. One indicates man, and on the other hand, zero indicates woman. Or when you have 2 one-hot encoded columns which indicate 3 different skin colors. If you want to change the one-hot encoded columns to see the new prediction of the model you should use Categorical Judge.

On the other hand, assume one of the columns is `weight` in Kg, and you want to see the prediction of the model, if the weight of each person was 10 Kg more. That's a numerical Case, and you should use Numerical Judge.

## How to use Courthouse for TensorFlow
First of all, you need to install TensorFlow. TensorFlow is not in the required_packages of the package.
```bash
pip install tensorflow
```

The next step is to build and train a TensorFlow model.
```python
import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10,1))
```
Of course, this is only a simple model. You should build your own model based on your requirements. Then you should train the model. (of course, you can skip this part. Courthouse only need a model)

Now let's use Courthouse.

### Categorical Judge
There are four steps to use categorical judge for TensorFlow:

First: import statements
```python
from courthouse.tensorflow.judge import CategoricalJudge
from courthouse.utils.case import CategoricalCase
```

Second: Instantiate a judge from CategoricalJudge class.
```python
judge = CategoricalJudge()
```

Third: Define the case. There are actually 2 different cases you can define in categorical judge.

I. The case is binary. For example assume our `sex` column. If it is zero, it indicates woman and if it is one, it indicates man. This is a binary column. In this case:
```python
judge.case(
    data,
    change_from = CategoricalCase(name = "woman", column = 3, binary = 0),
    change_towards = CategoricalCase(name = "man")
)
```
What is happening here? In case method, you specify your data first, then two CategoricalCases. In categorical cases the name is mandatory (you should provide the name), but it can be anything you want. column indicates that which column is the sex column. In this case, the fourth column (counting from 0) of the dataset indicates sex. And at last, if you want to convert all zeros to one (that's exactly our case), specify binary as 0. So in this example, we want to find all the women in our dataset and change them into men to see whether the output of the model is different. You don't need to specify column and binary for the second CategoricalCase (actually you can, but the judge doesn't consider it).

II. The case is not binary. Assume our skin colors example. We have 2 columns: one column indicates if the person is black if it is one. The second column indicates that the person is black if it is one. And if both are zero, then the person is brown. Assume white is column number 2 and white is column number 3. To change the skin color from black to white:
```python
judge.case(
    data,
    change_from = CategoricalCase(name = "black", column = 2),
    change_towards = CategoricalCase(name = "white", column = 3)
)
```
To change the skin color from white to brown:
```python
judge.case(
    data,
    change_from = CategoricalCase(name = "white", column = 3),
    change_towards = CategoricalCase(name = "brown")
)
```
And to change the skin color from brown to black:
```python
judge.case(
    data,
    change_from = CategoricalCase(name = "brown", column = [2,3]),
    change_towards = CategoricalCase(name = "black", column = 2)
)
```

Third: Judge the model. That's easy thing to do.
```python
judge.judge(model, output_type = "binary_sigmoid")
```
What is the output_type? That is suprisingly the output type of your model. output_type can be one of these four strings:

1. binary_sigmoid: when your output is binary and activation function is sigmoid.
2. binary_tanh: when your output is binary and activation function is tanh.
3. categorical: when your output is a multi-class classification
4. regression: when your model predicts a single number for a regression problem.

Fourth: Display the verdict of the judge.
```python
judge.verdict()
```

That's all. If you want to see which dataponit faced discrimination when you changed one specific attribute you can use the faced_discrimination method.
```python
judge.faced_discrminiation()
```
This method outputs a dictionary of datapoints which encountered discrimination. Keys are just simple indices, and values are actual datapoints.

### Numerical Judge
Using numerical judge for TensorFlow is even easier than using the categorical judge.
```python
from courthouse.tensorflow.judge import NumericalJudge
from courthouse.utils.case import NumericalCase

judge = NumericalJudge()
judge.case(
    data,
    NumericalCase(name = "weight", column = 4),
    change_amount = 30
)

judge.judge(model, output_type = "binary_sigmoid")
judge.verdict()
```
and optionaly
```python
faced_discrimination = judge.faced_discrimination()
```
In the example above, we defined a numerical judge, and wanted to judge the model based on increasing the weight for 30 (kg). That's it.

### Important Note
You can not use faced_discrimination method when the output_type of the model is regression.

## How to use Courthouse for PyTorch
Using Courthouse for PyTorch is as easy as using it for TensorFlow. Just two differences.
1. import statement
```python
from courthouse.pytorch.judge import CategoricalJudge, NumericalJudge
```
2. You also can use `binary_with_logits` as output_type of judge method.

## Contribution
I would be glad of any contribution to this package. Here is my LinkedIn account if you want to be in touch: https://www.linkedin.com/in/arashheidari .