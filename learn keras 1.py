#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# <h1>input the shape of one input:</h1><br>the following example shape will be<h3>(none ,784)</h3>

# In[2]:


inputs = keras.Input(shape=(784,))
print(inputs.shape)
print(inputs.dtype)


# dense layer is : <b>activition_function(np.dot(x,w)+bias)</b>

# In[3]:


dense = layers.Dense(64, activation="relu")
x = dense(inputs)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)


# In[6]:


model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
model.summary()


# In[7]:


keras.utils.plot_model(model, "my_first_model.png")


# In[16]:


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape)
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])


# In[17]:


"""This saved file includes the:
-model architecture
-model weight values (that were learned during training)
-model training config, if any (as passed to compile)
-optimizer and its state, if any (to restart training where you left off)"""
model.save("path_to_my_model")
del model


# In[ ]:


# Recreate the exact same model purely from the file:
model = keras.models.load_model("path_to_my_model")

