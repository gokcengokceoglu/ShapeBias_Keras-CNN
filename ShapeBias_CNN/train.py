import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"         # 0 for GPU

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from model import Model, load_model

img_width, img_height = 200, 200
batch_size = 4
epoch_num = 50

model_obj = Model()
model = model_obj.get_model()
model.summary()
from keras.utils import plot_model
plot_model(model, to_file='model.png')
## AUGMENTATION CONFIGURATION FOR TRAINING ##
train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'train_data/train_data',  # (for example, 'C:\\Train')m
        target_size=(200,200),
        color_mode='rgb',

        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'val_data/val_data',
        target_size=(200, 200),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical')

filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)


tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=True)
callbacks_list = [checkpoint, tensorboard]

model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epoch_num,
        callbacks=callbacks_list,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size)

np.save('class_indices.npy', train_generator.class_indices)
model.save('model.h5')
model.save_weights('model_weights.h5')