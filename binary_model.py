from collections import Counter
import humanize
import tensorflow as tf
import numpy as np
from datetime import datetime
from keras.layers import Conv2D, BatchNormalization, ReLU, Dropout, Dense, \
    MaxPooling2D, Input, UpSampling2D, concatenate, Flatten
from keras.models import Sequential
from sklearn.utils import class_weight
from imblearn.keras import balanced_batch_generator
from imblearn.under_sampling import NearMiss
from keras_preprocessing.image import DirectoryIterator
from keras.callbacks import  TensorBoard

class ConvModel:
    def __init__(self, input_size=(1024, 1024, 1)):
        model = Sequential()
        self.input_size = input_size
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_size))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(2048, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print(model.summary())
        self.model = model

    def fit(self, data_gen:DirectoryIterator, labels):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.keras.backend.set_session(tf.Session(config=config))
        class_weights = class_weight.compute_class_weight('balanced', [1, 0],
                                                          [int(x[0][0] != -1) for x in labels.values()])
        data_labels = (int(x[0][0] != -1) for x in labels.values())
        # labels_ = []
        # for v in labels.values():
        #     labels.append(int(v['mask'][0][0] != -1))
        # print("Classes are: ", Counter(labels))
        # imb_gen, steps = balanced_batch_generator(data_gen, data_labels, batch_size=data_gen.batch_size, sampler=NearMiss())
        tb = TensorBoard(log_dir='/tmp/logs')

        self.model.fit_generator(data_gen, epochs=10, class_weight=class_weights, callbacks=[tb])

    def predict(self, image) -> int:
        img_array = np.array(image).reshape((1,) + self.input_size)
        return self.model.predict(img_array)

    def save(self):
        return self.model.save(f'models/convnet_{int(datetime.now().timestamp())}.h5')
