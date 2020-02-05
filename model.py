from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda
from keras.models import load_model
from keras.models import Model as keras_model
from keras import backend as K

class Model:
    """
    Attributes:
        channel_num: representing the color mode. channel_num=1 for grayscale, channel_num=3 for rgb images.
    """

    def __init__(self, img_width=200, img_height=200, color_mode=0):
        self.img_width = img_width
        self.img_height = img_height

        if color_mode:
            self.channel_num = 1
        else:
            self.channel_num = 3

    def get_model(self):

        if K.image_data_format() == 'channels_first':
            input_shape = (self.channel_num, self.img_width, self.img_height)
        else:
            input_shape = (self.img_width, self.img_height, self.channel_num)

        # MODEL CREATION #
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3),activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu',))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu',))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu',))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(256, activation='relu',))
        model.add(Dropout(0.5))
        model.add(Dense(5))
        model.add(Lambda(lambda x: K.tf.nn.softmax(x)))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print(model.summary())

        return model

    '''
    Freeze the pre-trained model's layers except the last 'layer_num' layers
    Return value: model
    '''
    def get_finetuned_model(self, path_to_pretrained_model, layer_num=0, layer_name='max_pooling2d_5'):

        pre_trained_model = load_model(path_to_pretrained_model)
        model_top_removed = keras_model(inputs=pre_trained_model.input, outputs=pre_trained_model.get_layer(layer_name).output)

        # Freeze the layers except the last 'layer_num' layers
        for layer in model_top_removed.layers[:-layer_num]:
            layer.trainable = False

        # Check the trainable status of the individual layers
        for layer in model_top_removed.layers:
            print('Checking the trainable status of the individual layers...')
            print(layer, layer.trainable)

        # Create the model
        model = Sequential()
        model.add(model_top_removed)

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print(model.summary())

        return model

