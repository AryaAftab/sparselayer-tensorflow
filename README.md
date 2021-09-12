## Sparse Layer - Tensorflow
An implementation of Sparse Layers in tensorflow 2.x. 
Implementation of layers of Dense and Conv2D has been done. Other layers will be added.

## Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AryaAftab/sparselayer-tensorflow/blob/master/demo/sparselayer_tensorflow_demo.ipynb)
## Install

```bash
$ pip install sparselayer-tensorflow
```

## Usage

### Sparse Convolution Network with Sparse Fully Connected on Head
```python
import tensorflow as tf
from tensorflow.keras.layers import Input, ReLU, BatchNormalization, Flatten, MaxPool2D
from sparselayer_tensorflow import SparseLayerConv2D, SparseLayerDense

# Create Convolution Network
X = tf.keras.layers.Input(shape=(28, 28, 1))
x = SparseLayerConv2D(n_filters=32, density=0.5, filter_size=(3,3), stride=(1,1), padding='SAME')(X)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D((2,2))(x)

x = SparseLayerConv2D(n_filters=64, density=0.5, filter_size=(3,3), stride=(1,1), padding='SAME')(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D((2,2))(x)

x = Flatten()(x)

# Added Sparse Dense
y = SparseLayerDense(units=10, density=0.2, activation=tf.nn.softmax)(x)

model = tf.keras.models.Model(X, y)


# Hyperparameters
batch_size=256
epochs=30

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),  # Utilize optimizer
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

# Train the network
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    validation_split=0.1,
    epochs=epochs)
```