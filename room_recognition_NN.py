import os
import cv2
import imghdr
from PIL import Image
import tensorflow as tf
from matplotlib import patches
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import time
from ultralytics import YOLO
from tensorflow.python.keras.models import load_model

def training(data_dir):
    # Image extensions to be considered
    image_exts = ['jpeg', 'jpg', 'bmp', 'png']

    # Loop through image classes in the directory
    for image_class in os.listdir(data_dir):
        # Loop through images in each class
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try:
                # Reading the image and determining its type
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                # If the type is not in the allowed list, delete the image
                if tip not in image_exts:
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e:
                # Print an error message if the image causes issues
                print('Issue with image {}'.format(image_path))
                # Uncomment the following line to remove the image in case of errors
                # os.remove(image_path)

    # Creating a dataset from images in the directory
    data = tf.keras.utils.image_dataset_from_directory('data', shuffle=True, seed=1,
                                                       labels='inferred',
                                                       label_mode='categorical')
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()

    # Applying random horizontal flips and normalizing pixel values
    data = data.map(lambda x, y: (tf.image.random_flip_left_right(x / 255), y))
    data.as_numpy_iterator().next()

    # Splitting the dataset into training, validation, and test sets
    train_size = int(len(data) * .7)
    val_size = int(len(data) * .2)
    test_size = int(len(data) * .1)
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)

    # Defining the neural network model
    model = Sequential()

    # Convolutional layer with 16 filters of size (3, 3), ReLU activation function
    # Input image size (256, 256, 3)
    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))

    # Pooling layer to reduce dimensionality
    model.add(MaxPooling2D())

    # Convolutional layer with 32 filters of size (3, 3), ReLU activation function
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))

    # Pooling layer to reduce dimensionality
    model.add(MaxPooling2D())

    # Convolutional layer with 16 filters of size (3, 3), ReLU activation function
    model.add(Conv2D(16, (3, 3), 1, activation='relu'))

    # Pooling layer to reduce dimensionality
    model.add(MaxPooling2D())

    # Flattening layer to prepare data before the fully connected layer
    model.add(Flatten())

    # Fully connected layer with 256 neurons, ReLU activation function, and L2 regularization
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))

    # Output layer with 4 neurons (number of classes) and sigmoid activation function
    model.add(Dense(4, activation='sigmoid'))

    # Compiling the model with the Adam optimizer, Categorical Crossentropy loss function, and accuracy metric
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Displaying the model structure
    model.summary()

    # Setting up TensorBoard and early stopping
    logdir = 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Training the model
    hist = model.fit(train, epochs=30, validation_data=val,
                     callbacks=[tensorboard_callback, early_stopping])

    # Plotting loss and accuracy graphs
    fig = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()
    fig = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    return model


# List of rooms for classification
rooms_array = ['bathroom', 'bedroom', 'kitchen', 'office']

print('Do you want to train NN? Y/n')
answer = input()
if answer == 'Y':
    print('Enter name of new NN-model:')
    name = input()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    new_model = training('data')
    new_model.save(os.path.join('models', '{}.h5'.format(name)))
    print('{}.h5 saved to folder `models` ')

print('Do you want to load existing NN? Y/n')
answer = input()
recognition = True
if answer == 'Y':
    print('Enter name of NN-model:')
    name = input()
    model = load_model(os.path.join('models', name), compile=False)
    while recognition:
        print('Do you want to recognise any image? Y/n')
        answer = input()
        if answer == 'Y':
            print('Enter name of the image you want to recognise:')
            image_name = input()
            result = model.predict(np.expand_dims(tf.image.resize(cv2.imread(image_name),
                                                                  (256, 256))/ 255, 0)).tolist()[0]
            # Determining the maximum value and its corresponding index
            max_value = max(result, default=None)
            max_index = result.index(max_value) if max_value is not None else -1

            # Outputting the classification result
            print('The room is ', rooms_array[max_index])
            print('The probabilities in raw: \n', result)

        else:
            recognition = False


