# Courthouse
Courthouse is a AI fairness analysis library for machine learning models. This model supports both TensorFlow and PyTorch.

## Installation
To install the library use pip.
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

For example, assume that your dataset has a `sex` column. If the columns is one, it indicates man, and zero indicates woman. Or when you have 2 one-hot encoded columns which indicate 3 different races. If you want to change the one-hot encoded columns to see the new prediction of the model you should use Categorical Judge.

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
Of course, this is only a simple model. You should build your own model based on your requirements. Then you should train the model. (of course you can skip this part. Courthouse only need a model)

Now let's use Courthouse. Let's see the full example first, and then I explain the vital parts.
```python
from courthouse.tensorflow.judge import CategoricalJudge
from courthouse.utils.case import CategircalCase

judge = CategoricalJudge()
judge.case(
    data,
    CategoricalCase(name = "woman", column = 2),
    CategoricalCase(name = "man")
)
judge.judge(model, output_type = "binary_tanh")
judge.verdict()
faced_discrimination = judge.faced_discrimination()
```
The four steps are:

1. Instantiate a Categorical Judge
2. define the case: In this case we are saying to Courthouse that we want to see if changing the sex from woman to man has an effect on the prediction of the model. We indicate that the one-hot encoded for sex is column number 2 (counting from 0).