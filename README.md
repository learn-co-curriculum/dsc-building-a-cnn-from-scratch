
# Building a CNN from Scratch - Lab

## Introduction

Now that you have background knowledge regarding how CNNs work and how to implement them via Keras, its time to practice those skills a little more independently in order to build a CNN on your own to solve a image recognition problem. In this lab, you'll practice building an image classifier from start to finish using a CNN.  

## Objectives

You will be able to:
* Transform images into tensors
* Build a CNN model for image recognition

## Loading the Images

The data for this lab concerns classifying lung xray images for pneumonia. The original dataset is from kaggle. We have downsampled this dataset in order to reduce training time for you when you design and fit your model to the data. It is anticipated that this process will take approximately 1 hour to run on a standard machine, although times will vary depending on your particular computer and set up. At the end of this lab, you are welcome to try training on the complete dataset and observe the impact on the model's overall accuracy. 

You can find the initial downsampled dataset in a subdirectory, **chest_xray**, of this repository.


```python
#Your code here; load the images; be sure to also preprocess these into tensors.
```


```python
# __SOLUTION__ 
from keras.preprocessing.image import ImageDataGenerator
import datetime

original_start = datetime.datetime.now()
start = datetime.datetime.now()
```

    /Users/matthew.mitchell/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.



```python
# __SOLUTION__ 
train_dir = 'chest_xray_downsampled/train'
validation_dir = 'chest_xray_downsampled/val/'
test_dir = 'chest_xray_downsampled/test/'

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
```

    Found 1738 images belonging to 2 classes.
    Found 4 images belonging to 2 classes.


## Designing the Model

Now it's time to design your CNN! Remember a few things when doing this: 
* You should alternate convolutional and pooling layers
* You should have later layers have a larger number of parameters in order to detect more abstract patterns
* Add some final dense layers to add a classifier to the convolutional base


```python
#Your code here; design and compile the model
```


```python
# __SOLUTION__ 
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


```python
# __SOLUTION__ 
from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
```

## Training and Evaluating the Model

Remember that training deep networks is resource intensive: depending on the size of the data, even a CNN with 3-4 successive convolutional and pooling layers is apt to take a hours to train on a high end laptop. Using 30 epochs and 8 layers (alternating between convolutional and pooling), our model took about 40 minutes to run on a year old macbook pro.


If you are concerned with runtime, you may want to set your model to run the training epochs overnight.  

**If you are going to run this process overnight, be sure to also script code for the following questions concerning data augmentation. Check your code twice (or more) and then set the notebook to run all, or something equivalent to have them train overnight.**


```python
#Set the model to train; see warnings above
```


```python
# Plot history
```


```python
# __SOLUTION__ 
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)
```

    Epoch 1/30
    100/100 [==============================] - 86s 855ms/step - loss: 0.4887 - acc: 0.7670 - val_loss: 0.8535 - val_acc: 0.7500
    Epoch 2/30
    100/100 [==============================] - 81s 814ms/step - loss: 0.2769 - acc: 0.8855 - val_loss: 0.8954 - val_acc: 0.7500
    Epoch 3/30
    100/100 [==============================] - 83s 828ms/step - loss: 0.1883 - acc: 0.9295 - val_loss: 0.8353 - val_acc: 0.7500
    Epoch 4/30
    100/100 [==============================] - 82s 815ms/step - loss: 0.1391 - acc: 0.9434 - val_loss: 1.0113 - val_acc: 0.7500
    Epoch 5/30
    100/100 [==============================] - 82s 821ms/step - loss: 0.1257 - acc: 0.9509 - val_loss: 1.0028 - val_acc: 0.7500
    Epoch 6/30
    100/100 [==============================] - 81s 809ms/step - loss: 0.1099 - acc: 0.9525 - val_loss: 0.9202 - val_acc: 0.7500
    Epoch 7/30
    100/100 [==============================] - 79s 786ms/step - loss: 0.0905 - acc: 0.9649 - val_loss: 1.2020 - val_acc: 0.7500
    Epoch 8/30
    100/100 [==============================] - 78s 781ms/step - loss: 0.0873 - acc: 0.9699 - val_loss: 1.2455 - val_acc: 0.7500
    Epoch 9/30
    100/100 [==============================] - 78s 779ms/step - loss: 0.0733 - acc: 0.9740 - val_loss: 0.8685 - val_acc: 0.7500
    Epoch 10/30
    100/100 [==============================] - 78s 779ms/step - loss: 0.0704 - acc: 0.9730 - val_loss: 0.5022 - val_acc: 0.7500
    Epoch 11/30
    100/100 [==============================] - 77s 771ms/step - loss: 0.0641 - acc: 0.9770 - val_loss: 3.1135 - val_acc: 0.5000
    Epoch 12/30
    100/100 [==============================] - 78s 784ms/step - loss: 0.0612 - acc: 0.9759 - val_loss: 0.8625 - val_acc: 0.7500
    Epoch 13/30
    100/100 [==============================] - 78s 783ms/step - loss: 0.0504 - acc: 0.9820 - val_loss: 0.5369 - val_acc: 0.7500
    Epoch 14/30
    100/100 [==============================] - 79s 786ms/step - loss: 0.0426 - acc: 0.9849 - val_loss: 1.5661 - val_acc: 0.7500
    Epoch 15/30
    100/100 [==============================] - 78s 782ms/step - loss: 0.0390 - acc: 0.9860 - val_loss: 0.8814 - val_acc: 0.7500
    Epoch 16/30
    100/100 [==============================] - 79s 794ms/step - loss: 0.0365 - acc: 0.9865 - val_loss: 2.3576 - val_acc: 0.7500
    Epoch 17/30
    100/100 [==============================] - 78s 781ms/step - loss: 0.0277 - acc: 0.9885 - val_loss: 3.1242 - val_acc: 0.5000
    Epoch 18/30
    100/100 [==============================] - 87s 866ms/step - loss: 0.0274 - acc: 0.9915 - val_loss: 2.1897 - val_acc: 0.7500
    Epoch 19/30
    100/100 [==============================] - 90s 895ms/step - loss: 0.0261 - acc: 0.9915 - val_loss: 1.5764 - val_acc: 0.7500
    Epoch 20/30
    100/100 [==============================] - 89s 891ms/step - loss: 0.0215 - acc: 0.9920 - val_loss: 2.8503 - val_acc: 0.5000
    Epoch 21/30
    100/100 [==============================] - 89s 893ms/step - loss: 0.0234 - acc: 0.9925 - val_loss: 0.5651 - val_acc: 0.7500
    Epoch 22/30
    100/100 [==============================] - 88s 883ms/step - loss: 0.0187 - acc: 0.9924 - val_loss: 1.5322 - val_acc: 0.7500
    Epoch 23/30
    100/100 [==============================] - 89s 887ms/step - loss: 0.0154 - acc: 0.9955 - val_loss: 0.8898 - val_acc: 0.7500
    Epoch 24/30
    100/100 [==============================] - 90s 904ms/step - loss: 0.0130 - acc: 0.9960 - val_loss: 1.2518 - val_acc: 0.7500
    Epoch 25/30
    100/100 [==============================] - 88s 882ms/step - loss: 0.0116 - acc: 0.9960 - val_loss: 0.3684 - val_acc: 0.7500
    Epoch 26/30
    100/100 [==============================] - 88s 877ms/step - loss: 0.0116 - acc: 0.9955 - val_loss: 1.5874 - val_acc: 0.7500
    Epoch 27/30
    100/100 [==============================] - 88s 879ms/step - loss: 0.0112 - acc: 0.9975 - val_loss: 1.5677 - val_acc: 0.7500
    Epoch 28/30
    100/100 [==============================] - 89s 886ms/step - loss: 0.0139 - acc: 0.9959 - val_loss: 0.7602 - val_acc: 0.7500
    Epoch 29/30
    100/100 [==============================] - 89s 892ms/step - loss: 0.0054 - acc: 0.9985 - val_loss: 2.1922 - val_acc: 0.7500
    Epoch 30/30
    100/100 [==============================] - 89s 895ms/step - loss: 0.0081 - acc: 0.9989 - val_loss: 2.8709 - val_acc: 0.5000



```python
# __SOLUTION__ 
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


![png](index_files/index_13_0.png)



![png](index_files/index_13_1.png)



```python
# __SOLUTION__ 
end = datetime.datetime.now()
elapsed = end - start
print('Training took a total of {}'.format(elapsed))
```

    Training took a total of 0:41:49.421757


## Save the Model


```python
#Your code here; save the model for future reference.
```


```python
# __SOLUTION__ 
model.save('chest_xray_downsampled_data.h5')
```

## Data Augmentation

Recall that data augmentation is typically always a necessary step when using a small dataset as this one which you have been provided. As such, if you haven't already, implement a data augmentation setup.

**Warning: This process took nearly 4 hours to run on a relatively new macbook pro. As such, it is recommended that you simply code the setup and compare to the solution branch, or set the process to run overnight if you do choose to actually run the code.**


```python
#Add data augmentation to the model setup and set the model to train; 
#See warnings above if you intend to run this block of code
```


```python
# __SOLUTION__ 
start = datetime.datetime.now()
```


```python
# __SOLUTION__ 
train_datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)
```

    Found 1738 images belonging to 2 classes.
    Found 4 images belonging to 2 classes.
    Epoch 1/100
    100/100 [==============================] - 142s 1s/step - loss: 4.0534 - acc: 0.7458 - val_loss: 6.0960 - val_acc: 0.5000
    Epoch 2/100
    100/100 [==============================] - 141s 1s/step - loss: 3.9857 - acc: 0.7500 - val_loss: 6.0960 - val_acc: 0.5000
    Epoch 3/100
    100/100 [==============================] - 141s 1s/step - loss: 4.2714 - acc: 0.7321 - val_loss: 6.0960 - val_acc: 0.5000
    Epoch 4/100
    100/100 [==============================] - 141s 1s/step - loss: 4.0029 - acc: 0.7489 - val_loss: 6.0960 - val_acc: 0.5000
    Epoch 5/100
    100/100 [==============================] - 139s 1s/step - loss: 4.0745 - acc: 0.7444 - val_loss: 6.0960 - val_acc: 0.5000
    Epoch 6/100
    100/100 [==============================] - 140s 1s/step - loss: 4.0733 - acc: 0.7445 - val_loss: 6.0960 - val_acc: 0.5000
    Epoch 7/100
    100/100 [==============================] - 138s 1s/step - loss: 4.1377 - acc: 0.7405 - val_loss: 6.0960 - val_acc: 0.5000
    Epoch 8/100
    100/100 [==============================] - 139s 1s/step - loss: 4.0505 - acc: 0.7459 - val_loss: 6.0960 - val_acc: 0.5000
    Epoch 9/100
    100/100 [==============================] - 140s 1s/step - loss: 3.9229 - acc: 0.7540 - val_loss: 3.9954 - val_acc: 0.5000
    Epoch 10/100
    100/100 [==============================] - 141s 1s/step - loss: 4.1402 - acc: 0.7403 - val_loss: 7.7979 - val_acc: 0.5000
    Epoch 11/100
    100/100 [==============================] - 140s 1s/step - loss: 4.0920 - acc: 0.7433 - val_loss: 7.7979 - val_acc: 0.5000
    Epoch 12/100
    100/100 [==============================] - 141s 1s/step - loss: 4.1529 - acc: 0.7395 - val_loss: 7.7979 - val_acc: 0.5000
    Epoch 13/100
    100/100 [==============================] - 141s 1s/step - loss: 4.0495 - acc: 0.7460 - val_loss: 7.7979 - val_acc: 0.5000
    Epoch 14/100
    100/100 [==============================] - 144s 1s/step - loss: 3.9658 - acc: 0.7513 - val_loss: 7.7979 - val_acc: 0.5000
    Epoch 15/100
    100/100 [==============================] - 139s 1s/step - loss: 4.2289 - acc: 0.7347 - val_loss: 7.7979 - val_acc: 0.5000
    Epoch 16/100
    100/100 [==============================] - 138s 1s/step - loss: 4.0146 - acc: 0.7482 - val_loss: 7.7979 - val_acc: 0.5000
    Epoch 17/100
    100/100 [==============================] - 141s 1s/step - loss: 4.0923 - acc: 0.7433 - val_loss: 7.7979 - val_acc: 0.5000
    Epoch 18/100
    100/100 [==============================] - 140s 1s/step - loss: 4.0155 - acc: 0.7481 - val_loss: 7.7979 - val_acc: 0.5000
    Epoch 19/100
    100/100 [==============================] - 140s 1s/step - loss: 4.1624 - acc: 0.7389 - val_loss: 7.7979 - val_acc: 0.5000
    Epoch 20/100
    100/100 [==============================] - 140s 1s/step - loss: 4.0242 - acc: 0.7476 - val_loss: 5.0995 - val_acc: 0.5000
    Epoch 21/100
    100/100 [==============================] - 140s 1s/step - loss: 3.9707 - acc: 0.7509 - val_loss: 4.4620 - val_acc: 0.5000
    Epoch 22/100
    100/100 [==============================] - 142s 1s/step - loss: 4.0276 - acc: 0.7474 - val_loss: 4.4620 - val_acc: 0.5000
    Epoch 23/100
    100/100 [==============================] - 141s 1s/step - loss: 3.9362 - acc: 0.7529 - val_loss: 4.8149 - val_acc: 0.5000
    Epoch 24/100
    100/100 [==============================] - 139s 1s/step - loss: 3.8175 - acc: 0.7606 - val_loss: 4.8149 - val_acc: 0.5000
    Epoch 25/100
    100/100 [==============================] - 138s 1s/step - loss: 3.9082 - acc: 0.7543 - val_loss: 1.1326 - val_acc: 0.7500
    Epoch 26/100
    100/100 [==============================] - 140s 1s/step - loss: 3.7905 - acc: 0.7625 - val_loss: 4.3103 - val_acc: 0.5000
    Epoch 27/100
    100/100 [==============================] - 141s 1s/step - loss: 3.9443 - acc: 0.7526 - val_loss: 4.3103 - val_acc: 0.5000
    Epoch 28/100
    100/100 [==============================] - 143s 1s/step - loss: 4.0380 - acc: 0.7467 - val_loss: 4.3103 - val_acc: 0.5000
    Epoch 29/100
    100/100 [==============================] - 140s 1s/step - loss: 3.8289 - acc: 0.7596 - val_loss: 6.1703 - val_acc: 0.5000
    Epoch 30/100
    100/100 [==============================] - 142s 1s/step - loss: 4.0844 - acc: 0.7438 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 31/100
    100/100 [==============================] - 144s 1s/step - loss: 3.9489 - acc: 0.7523 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 32/100
    100/100 [==============================] - 141s 1s/step - loss: 4.0662 - acc: 0.7449 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 33/100
    100/100 [==============================] - 140s 1s/step - loss: 4.1484 - acc: 0.7398 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 34/100
    100/100 [==============================] - 144s 1s/step - loss: 4.1580 - acc: 0.7392 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 35/100
    100/100 [==============================] - 128s 1s/step - loss: 3.9530 - acc: 0.7520 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 36/100
    100/100 [==============================] - 128s 1s/step - loss: 4.2274 - acc: 0.7348 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 37/100
    100/100 [==============================] - 127s 1s/step - loss: 4.0704 - acc: 0.7447 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 38/100
    100/100 [==============================] - 127s 1s/step - loss: 4.1519 - acc: 0.7396 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 39/100
    100/100 [==============================] - 126s 1s/step - loss: 4.0574 - acc: 0.7455 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 40/100
    100/100 [==============================] - 126s 1s/step - loss: 4.0704 - acc: 0.7447 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 41/100
    100/100 [==============================] - 126s 1s/step - loss: 4.1102 - acc: 0.7422 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 42/100
    100/100 [==============================] - 127s 1s/step - loss: 4.1682 - acc: 0.7385 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 43/100
    100/100 [==============================] - 128s 1s/step - loss: 4.0685 - acc: 0.7448 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 44/100
    100/100 [==============================] - 128s 1s/step - loss: 4.0296 - acc: 0.7472 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 45/100
    100/100 [==============================] - 138s 1s/step - loss: 3.9958 - acc: 0.7494 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 46/100
    100/100 [==============================] - 139s 1s/step - loss: 4.1717 - acc: 0.7383 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 47/100
    100/100 [==============================] - 140s 1s/step - loss: 4.1433 - acc: 0.7401 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 48/100
    100/100 [==============================] - 140s 1s/step - loss: 4.0861 - acc: 0.7437 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 49/100
    100/100 [==============================] - 142s 1s/step - loss: 4.1052 - acc: 0.7425 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 50/100
    100/100 [==============================] - 142s 1s/step - loss: 4.0575 - acc: 0.7455 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 51/100
    100/100 [==============================] - 141s 1s/step - loss: 4.1219 - acc: 0.7415 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 52/100
    100/100 [==============================] - 140s 1s/step - loss: 4.1211 - acc: 0.7415 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 53/100
    100/100 [==============================] - 138s 1s/step - loss: 4.1011 - acc: 0.7428 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 54/100
    100/100 [==============================] - 137s 1s/step - loss: 4.1708 - acc: 0.7384 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 55/100
    100/100 [==============================] - 141s 1s/step - loss: 4.0237 - acc: 0.7476 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 56/100
    100/100 [==============================] - 141s 1s/step - loss: 4.0167 - acc: 0.7481 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 57/100
    100/100 [==============================] - 140s 1s/step - loss: 4.2275 - acc: 0.7348 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 58/100
    100/100 [==============================] - 141s 1s/step - loss: 4.1260 - acc: 0.7412 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 59/100
    100/100 [==============================] - 141s 1s/step - loss: 4.0205 - acc: 0.7478 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 60/100
    100/100 [==============================] - 140s 1s/step - loss: 4.1700 - acc: 0.7384 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 61/100
    100/100 [==============================] - 140s 1s/step - loss: 4.0983 - acc: 0.7429 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 62/100
    100/100 [==============================] - 139s 1s/step - loss: 4.1577 - acc: 0.7392 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 63/100
    100/100 [==============================] - 139s 1s/step - loss: 4.0762 - acc: 0.7443 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 64/100
    100/100 [==============================] - 140s 1s/step - loss: 4.0771 - acc: 0.7443 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 65/100
    100/100 [==============================] - 148s 1s/step - loss: 4.1686 - acc: 0.7385 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 66/100
    100/100 [==============================] - 144s 1s/step - loss: 4.1260 - acc: 0.7412 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 67/100
    100/100 [==============================] - 133s 1s/step - loss: 4.0990 - acc: 0.7429 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 68/100
    100/100 [==============================] - 134s 1s/step - loss: 4.1392 - acc: 0.7404 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 69/100
    100/100 [==============================] - 133s 1s/step - loss: 4.0504 - acc: 0.7459 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 70/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0721 - acc: 0.7446 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 71/100
    100/100 [==============================] - 141s 1s/step - loss: 4.1093 - acc: 0.7422 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 72/100
    100/100 [==============================] - 142s 1s/step - loss: 4.1588 - acc: 0.7391 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 73/100
    100/100 [==============================] - 131s 1s/step - loss: 4.0603 - acc: 0.7453 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 74/100
    100/100 [==============================] - 132s 1s/step - loss: 4.1301 - acc: 0.7409 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 75/100
    100/100 [==============================] - 133s 1s/step - loss: 4.0146 - acc: 0.7482 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 76/100
    100/100 [==============================] - 126s 1s/step - loss: 4.2000 - acc: 0.7366 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 77/100
    100/100 [==============================] - 126s 1s/step - loss: 4.0319 - acc: 0.7471 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 78/100
    100/100 [==============================] - 128s 1s/step - loss: 4.0217 - acc: 0.7477 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 79/100
    100/100 [==============================] - 127s 1s/step - loss: 4.1386 - acc: 0.7404 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 80/100
    100/100 [==============================] - 142s 1s/step - loss: 4.1061 - acc: 0.7424 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 81/100
    100/100 [==============================] - 144s 1s/step - loss: 4.1732 - acc: 0.7382 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 82/100
    100/100 [==============================] - 144s 1s/step - loss: 4.0970 - acc: 0.7430 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 83/100
    100/100 [==============================] - 145s 1s/step - loss: 4.1100 - acc: 0.7422 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 84/100
    100/100 [==============================] - 143s 1s/step - loss: 4.2066 - acc: 0.7361 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 85/100
    100/100 [==============================] - 142s 1s/step - loss: 4.1411 - acc: 0.7403 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 86/100
    100/100 [==============================] - 143s 1s/step - loss: 3.9498 - acc: 0.7522 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 87/100
    100/100 [==============================] - 144s 1s/step - loss: 4.1178 - acc: 0.7417 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 88/100
    100/100 [==============================] - 144s 1s/step - loss: 4.2025 - acc: 0.7364 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 89/100
    100/100 [==============================] - 144s 1s/step - loss: 4.1190 - acc: 0.7416 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 90/100
    100/100 [==============================] - 144s 1s/step - loss: 4.1284 - acc: 0.7410 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 91/100
    100/100 [==============================] - 143s 1s/step - loss: 4.0046 - acc: 0.7488 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 92/100
    100/100 [==============================] - 143s 1s/step - loss: 4.0744 - acc: 0.7444 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 93/100
    100/100 [==============================] - 142s 1s/step - loss: 4.1749 - acc: 0.7381 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 94/100
    100/100 [==============================] - 143s 1s/step - loss: 4.1058 - acc: 0.7425 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 95/100
    100/100 [==============================] - 142s 1s/step - loss: 4.1509 - acc: 0.7396 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 96/100
    100/100 [==============================] - 144s 1s/step - loss: 4.0820 - acc: 0.7440 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 97/100
    100/100 [==============================] - 144s 1s/step - loss: 4.0445 - acc: 0.7463 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 98/100
    100/100 [==============================] - 144s 1s/step - loss: 4.1733 - acc: 0.7382 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 99/100
    100/100 [==============================] - 145s 1s/step - loss: 4.0879 - acc: 0.7436 - val_loss: 7.9712 - val_acc: 0.5000
    Epoch 100/100
    100/100 [==============================] - 145s 1s/step - loss: 4.0949 - acc: 0.7431 - val_loss: 7.9712 - val_acc: 0.5000



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


![png](index_files/index_22_0.png)



![png](index_files/index_22_1.png)



```python
# __SOLUTION__ 
end = datetime.datetime.now()
elapsed = end - start
print('Training with data augmentation took a total of {}'.format(elapsed))
```

    Training with data augmentation took a total of 3:51:22.263853



```python
# __SOLUTION__ 
model.save('chest_xray_downsampled_with_augmentation_data.h5')
```

## Final Evaluation

Now use the test set to perform a final evaluation on your model of choice.


```python
# Your code here; perform a final evaluation using the test set..
```


```python
# __SOLUTION__ 
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)
```

    Found 208 images belonging to 2 classes.
    test acc: 0.6271008439675099


## Extension: Adding Data to the Model

As discussed, the current dataset we worked with is a subset of a dataset hosted on Kaggle. Increasing the data that we use to train the model will result in additional performance gains but will also result in longer training times and be more resource intensive.   

It is estimated that training on the full dataset will take approximately 4 hours (and potentially significantly longer) depending on your computer's specifications.

In order to test the impact of training on the full dataset, start by downloading the data from kaggle here: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia.   


```python
#Optional extension; Your code here
```


```python
# __SOLUTION__ 
#Optional extension; Your code here
# Imports from above-provided as reference
# from keras import optimizers
# from keras import layers
# from keras import models
# from keras.preprocessing.image import ImageDataGenerator
# import datetime
# import matplotlib.pyplot as plt
# %matplotlib inline 

start = datetime.datetime.now()

train_dir = 'chest_xray/train'
validation_dir = 'chest_xray/val/'
test_dir = 'chest_xray/test/'

#Basic Data Loading; All images will be rescaled by 1./255
# train_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#         # This is the target directory
#         train_dir,
#         # All images will be resized to 150x150
#         target_size=(150, 150),
#         batch_size=20,
#         # Since we use binary_crossentropy loss, we need binary labels
#         class_mode='binary')

# validation_generator = test_datagen.flow_from_directory(
#         validation_dir,
#         target_size=(150, 150),
#         batch_size=20,
#         class_mode='binary')

#With Data Augmentation
train_datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

#Never apply data augmentation to test/validation sets 
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

#Design Model Architecture
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


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)


#Viz

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

model.save('chest_xray_all_with_augmentation_data.h5')

end = datetime.datetime.now()
elapsed = end - start
print('Full data model training and evaluation took a total of:\n {}'.format(elapsed))
```

    Found 5216 images belonging to 2 classes.
    Found 16 images belonging to 2 classes.
    Epoch 1/100
    100/100 [==============================] - 155s 2s/step - loss: 4.1870 - acc: 0.7344 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 2/100
    100/100 [==============================] - 154s 2s/step - loss: 3.9956 - acc: 0.7494 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 3/100
    100/100 [==============================] - 155s 2s/step - loss: 4.1151 - acc: 0.7419 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 4/100
    100/100 [==============================] - 156s 2s/step - loss: 4.0653 - acc: 0.7450 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 5/100
    100/100 [==============================] - 156s 2s/step - loss: 4.1052 - acc: 0.7425 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 6/100
    100/100 [==============================] - 156s 2s/step - loss: 3.9607 - acc: 0.7516 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 7/100
    100/100 [==============================] - 156s 2s/step - loss: 4.2297 - acc: 0.7347 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 8/100
    100/100 [==============================] - 155s 2s/step - loss: 4.0753 - acc: 0.7444 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 9/100
    100/100 [==============================] - 155s 2s/step - loss: 4.1400 - acc: 0.7403 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 10/100
    100/100 [==============================] - 154s 2s/step - loss: 4.0603 - acc: 0.7453 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 11/100
    100/100 [==============================] - 154s 2s/step - loss: 4.0504 - acc: 0.7459 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 12/100
    100/100 [==============================] - 157s 2s/step - loss: 4.0902 - acc: 0.7434 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 13/100
    100/100 [==============================] - 156s 2s/step - loss: 4.1649 - acc: 0.7387 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 14/100
    100/100 [==============================] - 157s 2s/step - loss: 4.0553 - acc: 0.7456 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 15/100
    100/100 [==============================] - 156s 2s/step - loss: 4.1799 - acc: 0.7378 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 16/100
    100/100 [==============================] - 155s 2s/step - loss: 4.0504 - acc: 0.7459 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 17/100
    100/100 [==============================] - 154s 2s/step - loss: 4.2148 - acc: 0.7356 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 18/100
    100/100 [==============================] - 154s 2s/step - loss: 3.9906 - acc: 0.7497 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 19/100
    100/100 [==============================] - 156s 2s/step - loss: 4.1151 - acc: 0.7419 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 20/100
    100/100 [==============================] - 157s 2s/step - loss: 4.1749 - acc: 0.7381 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 21/100
    100/100 [==============================] - 158s 2s/step - loss: 4.1351 - acc: 0.7406 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 22/100
    100/100 [==============================] - 156s 2s/step - loss: 3.9806 - acc: 0.7503 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 23/100
    100/100 [==============================] - 158s 2s/step - loss: 4.1400 - acc: 0.7403 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 24/100
    100/100 [==============================] - 155s 2s/step - loss: 4.0703 - acc: 0.7447 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 25/100
    100/100 [==============================] - 155s 2s/step - loss: 4.0155 - acc: 0.7481 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 26/100
    100/100 [==============================] - 154s 2s/step - loss: 4.1400 - acc: 0.7403 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 27/100
    100/100 [==============================] - 156s 2s/step - loss: 4.0803 - acc: 0.7441 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 28/100
    100/100 [==============================] - 156s 2s/step - loss: 4.0852 - acc: 0.7438 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 29/100
    100/100 [==============================] - 156s 2s/step - loss: 4.1899 - acc: 0.7372 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 30/100
    100/100 [==============================] - 155s 2s/step - loss: 3.9906 - acc: 0.7497 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 31/100
    100/100 [==============================] - 156s 2s/step - loss: 4.1799 - acc: 0.7378 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 32/100
    100/100 [==============================] - 155s 2s/step - loss: 4.1251 - acc: 0.7412 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 33/100
    100/100 [==============================] - 155s 2s/step - loss: 4.0055 - acc: 0.7488 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 34/100
    100/100 [==============================] - 154s 2s/step - loss: 4.1649 - acc: 0.7387 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 35/100
    100/100 [==============================] - 156s 2s/step - loss: 4.0603 - acc: 0.7453 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 36/100
    100/100 [==============================] - 156s 2s/step - loss: 4.1301 - acc: 0.7409 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 37/100
    100/100 [==============================] - 157s 2s/step - loss: 4.1400 - acc: 0.7403 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 38/100
    100/100 [==============================] - 156s 2s/step - loss: 3.9507 - acc: 0.7522 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 39/100
    100/100 [==============================] - 157s 2s/step - loss: 4.1948 - acc: 0.7369 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 40/100
    100/100 [==============================] - 155s 2s/step - loss: 4.1151 - acc: 0.7419 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 41/100
    100/100 [==============================] - 155s 2s/step - loss: 4.1450 - acc: 0.7400 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 42/100
    100/100 [==============================] - 155s 2s/step - loss: 4.1699 - acc: 0.7384 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 43/100
    100/100 [==============================] - 155s 2s/step - loss: 4.0852 - acc: 0.7438 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 44/100
    100/100 [==============================] - 156s 2s/step - loss: 4.0005 - acc: 0.7491 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 45/100
    100/100 [==============================] - 141s 1s/step - loss: 4.0205 - acc: 0.7478 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 46/100
    100/100 [==============================] - 135s 1s/step - loss: 4.1151 - acc: 0.7419 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 47/100
    100/100 [==============================] - 137s 1s/step - loss: 4.2596 - acc: 0.7328 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 48/100
    100/100 [==============================] - 136s 1s/step - loss: 4.0255 - acc: 0.7475 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 49/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0653 - acc: 0.7450 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 50/100
    100/100 [==============================] - 135s 1s/step - loss: 4.1600 - acc: 0.7391 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 51/100
    100/100 [==============================] - 136s 1s/step - loss: 4.0155 - acc: 0.7481 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 52/100
    100/100 [==============================] - 135s 1s/step - loss: 4.1002 - acc: 0.7428 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 53/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0055 - acc: 0.7488 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 54/100
    100/100 [==============================] - 136s 1s/step - loss: 4.1699 - acc: 0.7384 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 55/100
    100/100 [==============================] - 135s 1s/step - loss: 4.1201 - acc: 0.7416 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 56/100
    100/100 [==============================] - 135s 1s/step - loss: 4.1899 - acc: 0.7372 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 57/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0255 - acc: 0.7475 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 58/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0155 - acc: 0.7481 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 59/100
    100/100 [==============================] - 135s 1s/step - loss: 4.2048 - acc: 0.7362 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 60/100
    100/100 [==============================] - 135s 1s/step - loss: 4.1151 - acc: 0.7419 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 61/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0952 - acc: 0.7431 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 62/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0205 - acc: 0.7478 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 63/100
    100/100 [==============================] - 136s 1s/step - loss: 4.2098 - acc: 0.7359 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 64/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0255 - acc: 0.7475 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 65/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0803 - acc: 0.7441 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 66/100
    100/100 [==============================] - 135s 1s/step - loss: 4.1749 - acc: 0.7381 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 67/100
    100/100 [==============================] - 136s 1s/step - loss: 4.0304 - acc: 0.7472 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 68/100
    100/100 [==============================] - 135s 1s/step - loss: 4.2198 - acc: 0.7353 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 69/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0653 - acc: 0.7450 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 70/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0803 - acc: 0.7441 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 71/100
    100/100 [==============================] - 135s 1s/step - loss: 4.1002 - acc: 0.7428 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 72/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0603 - acc: 0.7453 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 73/100
    100/100 [==============================] - 139s 1s/step - loss: 4.0852 - acc: 0.7438 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 74/100
    100/100 [==============================] - 136s 1s/step - loss: 4.1301 - acc: 0.7409 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 75/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0703 - acc: 0.7447 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 76/100
    100/100 [==============================] - 135s 1s/step - loss: 4.1849 - acc: 0.7375 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 77/100
    100/100 [==============================] - 134s 1s/step - loss: 3.9607 - acc: 0.7516 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 78/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0553 - acc: 0.7456 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 79/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0404 - acc: 0.7466 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 80/100
    100/100 [==============================] - 135s 1s/step - loss: 4.2496 - acc: 0.7334 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 81/100
    100/100 [==============================] - 135s 1s/step - loss: 4.1002 - acc: 0.7428 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 82/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0753 - acc: 0.7444 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 83/100
    100/100 [==============================] - 135s 1s/step - loss: 4.1052 - acc: 0.7425 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 84/100
    100/100 [==============================] - 135s 1s/step - loss: 4.1500 - acc: 0.7397 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 85/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0803 - acc: 0.7441 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 86/100
    100/100 [==============================] - 135s 1s/step - loss: 3.9906 - acc: 0.7497 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 87/100
    100/100 [==============================] - 135s 1s/step - loss: 4.1500 - acc: 0.7397 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 88/100
    100/100 [==============================] - 135s 1s/step - loss: 4.1500 - acc: 0.7397 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 89/100
    100/100 [==============================] - 136s 1s/step - loss: 4.1450 - acc: 0.7400 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 90/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0205 - acc: 0.7478 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 91/100
    100/100 [==============================] - 135s 1s/step - loss: 4.1600 - acc: 0.7391 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 92/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0404 - acc: 0.7466 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 93/100
    100/100 [==============================] - 135s 1s/step - loss: 4.1251 - acc: 0.7412 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 94/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0454 - acc: 0.7463 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 95/100
    100/100 [==============================] - 135s 1s/step - loss: 4.2048 - acc: 0.7362 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 96/100
    100/100 [==============================] - 135s 1s/step - loss: 4.0454 - acc: 0.7463 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 97/100
    100/100 [==============================] - 134s 1s/step - loss: 4.0504 - acc: 0.7459 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 98/100
    100/100 [==============================] - 135s 1s/step - loss: 4.1500 - acc: 0.7397 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 99/100
    100/100 [==============================] - 136s 1s/step - loss: 4.1052 - acc: 0.7425 - val_loss: 0.7459 - val_acc: 0.5000
    Epoch 100/100
    100/100 [==============================] - 136s 1s/step - loss: 4.0504 - acc: 0.7459 - val_loss: 0.7459 - val_acc: 0.5000



![png](index_files/index_30_1.png)



![png](index_files/index_30_2.png)


    Full data model training and evaluation took a total of:
     4:00:25.650249



```python
# __SOLUTION__ 
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)
```

    Found 624 images belonging to 2 classes.
    test acc: 0.8058943079739083



```python
# __SOLUTION__ 
end = datetime.datetime.now()
elapsed = end - original_start
print('Entire notebook took a total of:\n {}'.format(elapsed))
```

    Entire notebook took a total of:
     8:33:55.438275


## Summary

Well done! In this lab, you practice building your own CNN for image recognition which drastically outperformed our previous attempts using a standard deep learning model alone. In the upcoming sections, we'll continue to investigate further techniques associated with CNNs including visualizing the representations they learn and techniques to further bolster their performance when we have limited training data such as here.
