# Building a CNN from Scratch - Lab

## Introduction

Now that you have background knowledge regarding how Convolution Neural Networks (CNNs) work and how to build them using Keras, its time to practice those skills a little more independently in order to build a CNN (or ConvNet) on your own to solve a image recognition problem. In this lab, you'll practice building an image classifier from start to finish using a CNN.  

## Objectives

In this lab you will: 

- Load images from a hierarchical file structure using an image datagenerator 
- Apply data augmentation to image files before training a neural network 
- Build a CNN using Keras 
- Visualize and evaluate the performance of CNN models 

## Loading the Images

The data for this lab are a bunch of pictures of cats and dogs, and our task is to correctly classify a picture as one or the other. The [original dataset](https://www.kaggle.com/c/dogs-vs-cats) is from Kaggle. We have downsampled this dataset in order to reduce training time for you when you design and fit your model to the data. ⏰ It is anticipated that this process will take approximately one hour to run on a standard machine, although times will vary depending on your particular computer and set up. At the end of this lab, you are welcome to try training on the complete dataset and observe the impact on the model's overall accuracy. 

You can find the initial downsampled dataset in a subdirectory, **cats_dogs_downsampled**, of this repository. 


```python
# Load the images

train_dir = 'cats_dogs_downsampled/train'
validation_dir = 'cats_dogs_downsampled/val/'
test_dir = 'cats_dogs_downsampled/test/' 
```


```python
# __SOLUTION__
# Load the images

train_dir = 'cats_dogs_downsampled/train'
validation_dir = 'cats_dogs_downsampled/val/'
test_dir = 'cats_dogs_downsampled/test/' 
```


```python
# Set-up date time to track how long run time takes
import datetime

original_start = datetime.datetime.now()
start = datetime.datetime.now()
```


```python
# __SOLUTION__ 

import datetime

original_start = datetime.datetime.now()
start = datetime.datetime.now()
```


```python
# Preprocess the images into tensors
# Rescale the data by 1/.255 and use binary_crossentropy loss

```


```python
# __SOLUTION__ 
# Preprocess the images into tensors
# Rescale the data by 1/.255 and use binary_crossentropy loss

from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = val_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')
```

    Found 2140 images belonging to 2 classes.
    Found 420 images belonging to 2 classes.


## Designing the Model

Now it's time to design your CNN using Keras. Remember a few things when doing this: 

- You should alternate convolutional and pooling layers
- You should have later layers have a larger number of parameters in order to detect more abstract patterns
- Add some final dense layers to add a classifier to the convolutional base 
- Compile this model 


```python
# Design the model
# Note: You may get a comment from tf regarding your kernel. This is not a warning per se, but rather informational.
```


```python
# __SOLUTION__ 
# Design the model
# Note: You may get a comment from tf regarding your kernel. This is not a warning per se, but rather informational.

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

    Metal device set to: Apple M1 Pro


    2023-06-14 10:04:42.982381: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
    2023-06-14 10:04:42.982809: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)



```python
# Compile the model
```


```python
# __SOLUTION__ 
# Compile the model
from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=['acc'])
```

## Training and Evaluating the Model

Remember that training deep networks is resource intensive: depending on the size of the data, even a CNN with 3-4 successive convolutional and pooling layers is apt to take a hours to train on a high end laptop. See the code chunk below to see how long it took to run your model. 

If you are concerned with runtime, you may want to set your model to run the training epochs overnight.  

**If you are going to run this process overnight, be sure to also script code for the following questions concerning data augmentation. Check your code twice (or more) and then set the notebook to run all, or something equivalent to have them train overnight.** 


```python
# Set the model to train 
# Note: You may get a comment from tf regarding your GPU or sometning similar.
# This is not a warning per se, but rather informational.
# ⏰ This cell may take several minutes to run 

```


```python
# __SOLUTION__ 
# Set the model to train 
# Note: You may get a comment from tf regarding your GPU or sometning similar.
# This is not a warning per se, but rather informational.
# ⏰ This cell may take several minutes to run
history = model.fit(train_generator, 
                              steps_per_epoch=100, 
                              epochs=30, 
                              validation_data=validation_generator, 
                              validation_steps=20)
```

    Epoch 1/30


    2023-06-14 10:04:52.056247: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
    2023-06-14 10:04:52.275945: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    100/100 [==============================] - ETA: 0s - loss: 0.6897 - acc: 0.5375

    2023-06-14 10:04:55.448925: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    100/100 [==============================] - 4s 32ms/step - loss: 0.6897 - acc: 0.5375 - val_loss: 0.6812 - val_acc: 0.5075
    Epoch 2/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.6567 - acc: 0.6040 - val_loss: 0.6506 - val_acc: 0.5625
    Epoch 3/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.6055 - acc: 0.6775 - val_loss: 0.6607 - val_acc: 0.6000
    Epoch 4/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.5660 - acc: 0.7015 - val_loss: 0.5900 - val_acc: 0.6825
    Epoch 5/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.5402 - acc: 0.7320 - val_loss: 0.5672 - val_acc: 0.7175
    Epoch 6/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.5145 - acc: 0.7365 - val_loss: 0.5419 - val_acc: 0.7100
    Epoch 7/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.4923 - acc: 0.7645 - val_loss: 0.5527 - val_acc: 0.7100
    Epoch 8/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.4612 - acc: 0.7805 - val_loss: 0.5086 - val_acc: 0.7525
    Epoch 9/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.4393 - acc: 0.7885 - val_loss: 0.5498 - val_acc: 0.7225
    Epoch 10/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.4156 - acc: 0.8080 - val_loss: 0.5176 - val_acc: 0.7425
    Epoch 11/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.3895 - acc: 0.8270 - val_loss: 0.4893 - val_acc: 0.7575
    Epoch 12/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.3696 - acc: 0.8370 - val_loss: 0.5213 - val_acc: 0.7550
    Epoch 13/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.3458 - acc: 0.8465 - val_loss: 0.5115 - val_acc: 0.7475
    Epoch 14/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.3270 - acc: 0.8545 - val_loss: 0.5011 - val_acc: 0.7700
    Epoch 15/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.3005 - acc: 0.8740 - val_loss: 0.4847 - val_acc: 0.7650
    Epoch 16/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.2812 - acc: 0.8855 - val_loss: 0.4956 - val_acc: 0.7475
    Epoch 17/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.2527 - acc: 0.8890 - val_loss: 0.4930 - val_acc: 0.7750
    Epoch 18/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.2327 - acc: 0.9145 - val_loss: 0.5025 - val_acc: 0.7750
    Epoch 19/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.2102 - acc: 0.9235 - val_loss: 0.5303 - val_acc: 0.7725
    Epoch 20/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.1927 - acc: 0.9315 - val_loss: 0.5390 - val_acc: 0.7675
    Epoch 21/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.1705 - acc: 0.9360 - val_loss: 0.5840 - val_acc: 0.7550
    Epoch 22/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.1625 - acc: 0.9445 - val_loss: 0.5890 - val_acc: 0.7800
    Epoch 23/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.1377 - acc: 0.9535 - val_loss: 0.5965 - val_acc: 0.7600
    Epoch 24/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.1224 - acc: 0.9565 - val_loss: 0.6173 - val_acc: 0.7625
    Epoch 25/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.1095 - acc: 0.9605 - val_loss: 0.6362 - val_acc: 0.7750
    Epoch 26/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.0933 - acc: 0.9710 - val_loss: 0.6589 - val_acc: 0.7700
    Epoch 27/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.0755 - acc: 0.9810 - val_loss: 0.6991 - val_acc: 0.7650
    Epoch 28/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.0689 - acc: 0.9825 - val_loss: 0.7184 - val_acc: 0.7725
    Epoch 29/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.0607 - acc: 0.9800 - val_loss: 0.7673 - val_acc: 0.7600
    Epoch 30/30
    100/100 [==============================] - 3s 29ms/step - loss: 0.0511 - acc: 0.9875 - val_loss: 0.9144 - val_acc: 0.7500



```python
# Plot history
import matplotlib.pyplot as plt
%matplotlib inline

# Type code here for plot history
```


```python
# __SOLUTION__ 
# Plot history

import matplotlib.pyplot as plt
%matplotlib inline 

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
```


    
![png](index_files/index_16_0.png)
    



    
![png](index_files/index_16_1.png)
    



```python
# Check runtime

end = datetime.datetime.now()
elapsed = end - start
print('Training took a total of {}'.format(elapsed))
```


```python
# __SOLUTION__ 
# Check runtime

end = datetime.datetime.now()
elapsed = end - start
print('Training took a total of {}'.format(elapsed))
```

    Training took a total of 0:02:57.288249


## Save the Model


```python
# Save the model for future reference 
```


```python
# __SOLUTION__ 
# Save the model for future reference 

model.save('cats_dogs_downsampled_data.h5')
```

## Data Augmentation

Recall that data augmentation is typically always a necessary step when using a small dataset as this one which you have been provided. If you haven't already, implement a data augmentation setup.

**Warning: ⏰ This process may take awhile depending on your set-up. As such, make allowances for this as necessary.** 


```python
# Set-up date time to track how long run time takes
start = datetime.datetime.now()
```


```python
# __SOLUTION__ 
# Set-up date time to track how long run time takes
start = datetime.datetime.now()
```


```python
# Add data augmentation to the model setup and set the model to train; 
# See the warnings above if you intend to run these blocks of code 
# ⏰ These cells where may take quite some time to run
```


```python
# __SOLUTION__ 

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40, 
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True, 
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

history = model.fit(train_generator, 
                              steps_per_epoch=100, 
                              epochs=30, 
                              validation_data=validation_generator, 
                              validation_steps=20)
```

    Found 2140 images belonging to 2 classes.
    Epoch 1/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.7259 - acc: 0.6745 - val_loss: 0.5039 - val_acc: 0.7600
    Epoch 2/30
    100/100 [==============================] - 6s 56ms/step - loss: 0.5889 - acc: 0.6785 - val_loss: 0.4906 - val_acc: 0.7750
    Epoch 3/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.5594 - acc: 0.7155 - val_loss: 0.4567 - val_acc: 0.7650
    Epoch 4/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.5646 - acc: 0.7005 - val_loss: 0.4761 - val_acc: 0.7550
    Epoch 5/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.5464 - acc: 0.7155 - val_loss: 0.4420 - val_acc: 0.7850
    Epoch 6/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.5497 - acc: 0.7235 - val_loss: 0.4245 - val_acc: 0.8075
    Epoch 7/30
    100/100 [==============================] - 6s 57ms/step - loss: 0.5395 - acc: 0.7175 - val_loss: 0.5365 - val_acc: 0.7050
    Epoch 8/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.5218 - acc: 0.7345 - val_loss: 0.4378 - val_acc: 0.7875
    Epoch 9/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.5258 - acc: 0.7345 - val_loss: 0.4556 - val_acc: 0.7800
    Epoch 10/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.5014 - acc: 0.7485 - val_loss: 0.4317 - val_acc: 0.7775
    Epoch 11/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.5169 - acc: 0.7310 - val_loss: 0.4260 - val_acc: 0.8100
    Epoch 12/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.5159 - acc: 0.7360 - val_loss: 0.4324 - val_acc: 0.8000
    Epoch 13/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.5187 - acc: 0.7410 - val_loss: 0.4231 - val_acc: 0.8075
    Epoch 14/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.5150 - acc: 0.7385 - val_loss: 0.4236 - val_acc: 0.8125
    Epoch 15/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.4995 - acc: 0.7525 - val_loss: 0.4303 - val_acc: 0.8050
    Epoch 16/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.4967 - acc: 0.7575 - val_loss: 0.4075 - val_acc: 0.8100
    Epoch 17/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.5067 - acc: 0.7490 - val_loss: 0.4333 - val_acc: 0.8125
    Epoch 18/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.4930 - acc: 0.7585 - val_loss: 0.4701 - val_acc: 0.7750
    Epoch 19/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.4940 - acc: 0.7415 - val_loss: 0.4031 - val_acc: 0.8200
    Epoch 20/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.4871 - acc: 0.7660 - val_loss: 0.4960 - val_acc: 0.7600
    Epoch 21/30
    100/100 [==============================] - 6s 59ms/step - loss: 0.4747 - acc: 0.7720 - val_loss: 0.4089 - val_acc: 0.8300
    Epoch 22/30
    100/100 [==============================] - 6s 59ms/step - loss: 0.4858 - acc: 0.7570 - val_loss: 0.3874 - val_acc: 0.8150
    Epoch 23/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.4839 - acc: 0.7710 - val_loss: 0.4211 - val_acc: 0.8175
    Epoch 24/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.4683 - acc: 0.7640 - val_loss: 0.4130 - val_acc: 0.8125
    Epoch 25/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.4865 - acc: 0.7515 - val_loss: 0.4076 - val_acc: 0.8225
    Epoch 26/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.4616 - acc: 0.7755 - val_loss: 0.4057 - val_acc: 0.8275
    Epoch 27/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.4615 - acc: 0.7790 - val_loss: 0.4173 - val_acc: 0.7900
    Epoch 28/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.4662 - acc: 0.7785 - val_loss: 0.4043 - val_acc: 0.8125
    Epoch 29/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.4768 - acc: 0.7715 - val_loss: 0.4186 - val_acc: 0.7975
    Epoch 30/30
    100/100 [==============================] - 6s 58ms/step - loss: 0.4573 - acc: 0.7850 - val_loss: 0.4359 - val_acc: 0.7725



```python
# __SOLUTION__ 

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
```


    
![png](index_files/index_27_0.png)
    



    
![png](index_files/index_27_1.png)
    



```python
# Check runtime 

```


```python
# __SOLUTION__ 
# Check runtime
end = datetime.datetime.now()
elapsed = end - start
print('Training with data augmentation took a total of {}'.format(elapsed))
```

    Training with data augmentation took a total of 0:03:43.083708


Save the model for future reference.  


```python
# Save the model 

```


```python
# __SOLUTION__ 
# Save the model 
model.save('cats_dogs_downsampled_with_augmentation_data.h5')
```

## Final Evaluation

Now use the test set to perform a final evaluation on your model of choice. 


```python
# Perform a final evaluation using the test set
```


```python
# __SOLUTION__ 
# Perform a final evaluation using the test set

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(test_dir, 
                                                  target_size=(150, 150), 
                                                  batch_size=20, 
                                                  class_mode='binary')
test_loss, test_acc = model.evaluate(test_generator, steps=20)
print('test acc:', test_acc)
```

    Found 425 images belonging to 2 classes.
    20/20 [==============================] - 0s 17ms/step - loss: 0.4889 - acc: 0.7600
    test acc: 0.7599999904632568


## Summary

Well done. In this lab, you practice building your own CNN for image recognition which drastically outperformed our previous attempts using a standard deep learning model alone. In the upcoming sections, we'll continue to investigate further techniques associated with CNNs including visualizing the representations they learn and techniques to further bolster their performance when we have limited training data such as here.
