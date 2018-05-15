from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

img_width, img_height = 150, 150
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

input_shape = (img_width, img_height, 3) # 150x150x3
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape)) # 148x148x32
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # 74x74x32

model.add(Conv2D(32, (3, 3))) # 72x72x32
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # 36x36x32

model.add(Conv2D(64, (3, 3))) # 34x34x64
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # 17x17x64

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])
# model.summary()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples)

model.save_weights('first_try.h5')
