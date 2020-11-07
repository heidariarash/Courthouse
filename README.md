# Courthouse
Courthouse is a AI fairness analysis library for machine learning models. This model supports both TensorFlow and PyTorch.

## Installation
To install the library use pip.
```bash
pip install courthouse
```

## How to use Courthouse for TensorFlow
First of all, you need to install TensorFlow. TensorFlow is not in the required_packages of the package.
```bash
pip install tensorflow
```
The next step is to prepare you data for training the TensorFlow model(, and Courthouse). Keep that in mind, for Courthouse the data should be a numpy ndarray.
```python
import numpy as np
data = np.array(...)
```

The next step is to build and train a TensorFlow model.
```python
import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10,1))
```
Of course, this is only a simple model. You should build your own model based on your requirements. Then you should train the model. (of course you can skip this part. Courthouse only need a model)
