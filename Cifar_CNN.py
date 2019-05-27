from keras.datasets import cifar10
import numpy as np

np.random.seed(10)

(x_img_train, y_label_train), (x_img_test, y_label_test) = cifar10.load_data()

print('train:', len(x_img_train))
print('test:', len(x_img_test))
print(x_img_train.shape)
print(y_label_train.shape)

label_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
              5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

import matplotlib.pyplot as plt


def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25

    for i in range(0, num):
        ax = plt.subplot(5, 5, i+1)
        ax.imshow(images[idx], cmap='binary')
        title = str(i) + ',' + label_dict[labels[i][0]]
        if len(prediction) > 0:
            title += '=>' + label_dict[prediction[i]]
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()


# plot_images_labels_prediction(x_img_train, y_label_train, [], 0)

x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0


from keras.utils import np_utils

y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x_img_train_normalize, y_label_train_OneHot,
                          validation_split=0.2, epochs=10, batch_size=128, verbose=2)

from plot import show_train_history

show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')

scores = model.evaluate(x_img_test_normalize, y_label_test_OneHot)
print(scores[1])


prediction = model.predict_classes(x_img_test_normalize)

plot_images_labels_prediction(x_img_test, y_label_test, prediction, 0, 10)

Predicted_Probability = model.predict(x_img_test_normalize)


def show_Predicted_Probability(y, prediction, x_img, Predicted_Probability, i):
    print('label:', label_dict[y[i][0]], 'predict', label_dict[prediction[i]])
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(x_img_test[i], (32, 32, 3)))
    plt.show()
    for j in range(10):
        print(label_dict[j] + ' Probability:%1.9f' % (Predicted_Probability[i][j]))


show_Predicted_Probability(y_label_test, prediction, x_img_test, Predicted_Probability, 0)
show_Predicted_Probability(y_label_test, prediction, x_img_test, Predicted_Probability, 3)


import pandas as pd

y_label_test = y_label_test.reshape(-1)
print(label_dict)
print(pd.crosstab(y_label_test, prediction, rownames=['label'], colnames=['predict']))

