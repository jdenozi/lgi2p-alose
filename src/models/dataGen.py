
import os
import numpy
import numpy as np 
import pandas as pd
import sklearn
import random
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adadelta, Adagrad, Adamax
from tensorflow.keras.layers import Dense, Dropout, Flatten,Input
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, MaxPooling1D
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore", category=PendingDeprecationWarning)


inputShape=inputShape = (128,216,3)
batch_size = 2
training_path = "/media/Dossiers/denozi/Documents/Stage_Alose/image_directory/train/"
validation_path = "/media/Dossiers/denozi/Documents/Stage_Alose/image_directory/test/"

def add_noise(img):
    '''Add random noise to an image'''
    VARIABILITY = 50
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

class_weight = {0:1 , 1:5}

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.02,
        zoom_range=0.02,
        horizontal_flip=False)



test_datagen = ImageDataGenerator(rescale=1./255,
        preprocessing_function=add_noise)

train_generator = train_datagen.flow_from_directory(
        training_path,
        target_size=(128,216),
        batch_size=batch_size, class_mode='binary'
        )
nb_img_train = len(train_generator.classes)
validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size=(128,216),
        batch_size=batch_size,class_mode='binary'
        )

nb_img_test = len(validation_generator.classes)

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),activation='relu',input_shape=inputShape,padding='same'))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D())

model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer= Adam(lr=0.00001),
              metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.Recall()])


model.fit_generator(
        train_generator,
        steps_per_epoch=nb_img_train // batch_size,
        epochs=40,
        class_weight = class_weight,
        validation_data=validation_generator,
        validation_steps=nb_img_test // batch_size)

print(train_generator.class_indices)

current_dir,last_dir = os.path.split(os.getcwd())
model.save(current_dir+"/tmp/bulls_model/dataGen_CNN.h5")

test_steps_per_epoch = numpy.math.ceil(validation_generator.samples / validation_generator.batch_size)
pred = model.predict_generator(validation_generator, steps = test_steps_per_epoch)
predicted_classes = numpy.argmax(pred, axis=1)


nb_image_test = len(validation_generator)
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)

print(report)
