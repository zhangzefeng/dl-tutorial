import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
from keras.layers import Input,Conv2D,Flatten,Dense,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, 28, 28)
        return data

train_data = extract_data('../notMNIST-to-MNIST/train-images-idx3-ubyte.gz', 60000)
test_data = extract_data('../notMNIST-to-MNIST/t10k-images-idx3-ubyte.gz', 10000)


def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels

train_labels = extract_labels('../notMNIST-to-MNIST/train-labels-idx1-ubyte.gz',60000)
test_labels = extract_labels('../notMNIST-to-MNIST/t10k-labels-idx1-ubyte.gz',10000)

print("Training set (images) shape: {shape}".format(shape=train_data.shape))
print("Test set (images) shape: {shape}".format(shape=test_data.shape))

# Create dictionary of target classes
label_dict = {
 0: 'A',
 1: 'B',
 2: 'C',
 3: 'D',
 4: 'E',
 5: 'F',
 6: 'G',
 7: 'H',
 8: 'I',
 9: 'J',
}

# plt.figure(figsize=[5,5])
# 
# # Display the first image in training data
# plt.subplot(121)
# curr_img = np.reshape(train_data[0], (28,28))
# curr_lbl = train_labels[0]
# plt.imshow(curr_img, cmap='gray')
# plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
# 
# # Display the first image in testing data
# plt.subplot(122)
# curr_img = np.reshape(test_data[0], (28,28))
# curr_lbl = test_labels[0]
# plt.imshow(curr_img, cmap='gray')
# plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
# 
# plt.show()

train_data = train_data.reshape(-1, 28,28, 1)
test_data = test_data.reshape(-1, 28,28, 1)
print train_data.shape, test_data.shape
print train_data.dtype, test_data.dtype

train_data = train_data / np.max(train_data)
test_data = test_data / np.max(test_data)
print np.max(train_data), np.max(test_data)

from sklearn.model_selection import train_test_split
from keras.utils import np_utils
train_onehot = np_utils.to_categorical(train_labels)
print train_onehot.shape
train_X,valid_X,train_ground,valid_ground = train_test_split(train_data,
                                                             train_onehot, 
                                                             test_size=0.2, 
                                                             random_state=13)

noise_factor = 0.5
x_train_noisy = train_X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_X.shape)
x_valid_noisy = valid_X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=valid_X.shape)
x_test_noisy = test_data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=test_data.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_valid_noisy = np.clip(x_valid_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
#plt.figure(figsize=[5,5])
## Display the first image in training data
#plt.subplot(121)
#curr_img = np.reshape(x_train_noisy[1], (28,28))
#plt.imshow(curr_img, cmap='gray')
## Display the first image in testing data
#plt.subplot(122)
#curr_img = np.reshape(x_test_noisy[1], (28,28))
#plt.imshow(curr_img, cmap='gray')
#plt.show()

batch_size = 128
epochs = 50
inChannel = 1
x, y = 28, 28
input_img = Input(shape = (x, y, inChannel))

def autoencoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(input_img) #28 x 28 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(pool1) #14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(pool2) #7 x 7 x 128 (small and thick)

    #decoder
    conv4 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(conv3) #7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(up1) # 14 x 14 x 64
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
#    decoded = Conv2D(1, 3, 3, activation='sigmoid', border_mode='same')(up2) # 28 x 28 x 1
    flatten = Flatten()(up2)
    hidden = Dense(64)(flatten)
    cla = Dense(10, activation='softmax')(hidden)
    return cla

autoencoder = Model(input_img, autoencoder(input_img))
checkpoint="classifier.checkpoint"
load=checkpoint + "/checkpoint-03.hdf5"
if os.path.isfile(load):
    autoencoder.load_weights(load)
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
autoencoder.summary()

filepath=checkpoint + "/checkpoint-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
if not os.path.isfile(load):
    autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size, nb_epoch=epochs, verbose=1,validation_data=(valid_X, valid_ground), callbacks=[checkpoint])

pred = autoencoder.predict(test_data)
print pred.shape

import sys
# plt.figure(figsize=(20, 4))
# print("Test Images")
# for i in range(10):
#     plt.subplot(2, 10, i+1)
#     plt.imshow(test_data[i, ..., 0], cmap='gray')
#     curr_lbl = test_labels[i]
#     plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
# print
# plt.show()

print("test ground labels")
for i in range(100):
    sys.stdout.write(label_dict[test_labels[i]])
print

print("prediction")
for i in range(100):
    sys.stdout.write(label_dict[np.argmax(pred[i])])
print
