import numpy as np
#import cv2
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
images = []
labels = []
for i in range(43):
    path = os.path.join(os.getcwd(),'train',str(i))
    print(str(i)+"path")
    images_list = os.listdir(path)
    print("imagelist")
    for a in images_list:
        try:
            image = Image.open(path + '\\'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            images.append(image)
            labels.append(i)
        except:
            print("Error loading image")
images = np.array(images)
labels = np.array(labels)
image_train, image_test, label_train, label_test = train_test_split(images, labels, test_size=0.33,train_size=0.67,random_state=42)
label_train = to_categorical(label_train, 43)
label_test = to_categorical(label_test, 43)
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=image_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 17
result =model.fit(image_train, label_train, batch_size=64, epochs=epochs, validation_data=(image_test, label_test))
model.save("model.h5")