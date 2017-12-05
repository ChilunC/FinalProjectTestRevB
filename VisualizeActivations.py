from __future__ import print_function

import numpy as np
import keras

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from read_activations import get_activations, display_activations
import matplotlib
#matplotlib.use("gtk")
from matplotlib import pyplot as plt

from vis.visualization import visualize_saliency
from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations
#from IPython import get_ipython

batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax', name='preds'))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

a = get_activations(model, x_test[0:1], print_shape_only=True)  # with just one sample.

display_activations(a)



get_activations(model, x_test[0:200], print_shape_only=True)  # with 200 samples.

# Utility to search for layer index by name.
# Alternatively we can specify this as -1 since it corresponds to the last layer.
#layer_idx = -1#utils.find_layer_idx(model, 'preds')
#get_ipython().run_line_magic('matplotlib', 'inline')
# This corresponds to the Dense linear layer.
#for class_idx in np.arange(10):
#    indices = np.where(y_test[:, class_idx] == 1.)[0]
#    idx = indices[0]

#    f, ax = plt.subplots(1, 4)
#    ax[0].imshow(x_test[idx][..., 0])

#    for i, modifier in enumerate([None, 'guided', 'relu']):
        #grads = visualize_saliency(model, layer_idx, filter_indices=class_idx,
        #                           seed_input=x_test[idx], backprop_modifier=modifier)
#        grads = visualize_activation(model, layer_idx, filter_indices=class_idx,
#                                   seed_input=x_test[idx], backprop_modifier=modifier)
        #grads = visualize_activation(model, layer_idx, filter_indices=class_idx, seed_input=None, input_range=(0, 255), \
        #                            backprop_modifier=None, grad_modifier=None, act_max_weight=1, lp_norm_weight=10, \
        #                            tv_weight=10, **optimizer_params)

#        if modifier is None:
#            modifier = 'vanilla'
#        ax[i + 1].set_title(modifier)
#        ax[i + 1].imshow(grads, cmap='jet')


#plt.show()